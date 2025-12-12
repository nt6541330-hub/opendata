import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import BytesIO
from collections import Counter

import torch
from PIL import Image, ImageOps
from ultralytics import YOLO
from huggingface_hub import snapshot_download
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    pipeline,
)

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

# 引入项目公共模块
from common.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()

# ===== 配置参数 =====
LLM_MODEL = "qwen3:0.6b"
YOLO_CONF = 0.7
YOLO_IOU = 0.7
YOLO_IMGSZ = 800

# —— 模型位置（建议后续移入 config.settings）
MODEL_ROOT = Path("./yolo_models").resolve()
YOLO_MODEL_NAME = "yolov8s.pt"
YOLO_LOCAL_PATH = MODEL_ROOT / YOLO_MODEL_NAME

# 请确保服务器上这些路径存在，或者修改为相对路径
BLIP_LOCAL_DIR = Path("/blip/BLIP-main/blip1/")
BLIP_REPO_ID = "Salesforce/blip-image-captioning-base"

CLIP_LOCAL_DIR = Path("/blip/BLIP-main/models/clip-vit-large-patch14")
CLIP_REPO_ID = "openai/clip-vit-large-patch14"

# —— 设备参数
USE_CUDA = torch.cuda.is_available()
YOLO_DEVICE = "cuda" if USE_CUDA else "cpu"
BLIP_DEVICE = "cuda" if USE_CUDA else "cpu"
CLF_DEVICE = 0 if USE_CUDA else -1
BLIP_MAX_TOKENS = 30
BLIP_USE_PROMPT = True
CLF_TOPK = 1

# —— 场景分类
CLF_LABELS_EN = [
    "a busy airport terminal with passengers and airplanes",
    "a commercial seaport with ships, cranes and containers",
    "a person or a large crowd of people in a public place",
    "a natural landscape with mountains, trees or rivers",
]
CLF_LABEL_ZH_MAP = {
    "a busy airport terminal with passengers and airplanes": "机场场景",
    "a commercial seaport with ships, cranes and containers": "港口/码头场景",
    "a person or a large crowd of people in a public place": "人物/人群",
    "a natural landscape with mountains, trees or rivers": "自然风景",
}

IGNORE_CLASSES = {"remote", "vase", "tie"}

# —— LLM
try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain.schema.runnable import RunnableLambda

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# 全局模型句柄
yolo_mdl = None
blip_bundle: Optional[Dict[str, Any]] = None
llm = None
_CLF_PIPE: Optional[Any] = None


# =========================
# ===== 模型加载逻辑 =====
# =========================

def ensure_local_yolo_model(model_path: Path) -> Path:
    if not model_path.exists():
        logger.info(f"[YOLO] 未找到本地权重：{model_path}，尝试下载...")
        try:
            _ = YOLO(str(model_path))  # 尝试触发ultralytics下载
        except Exception:
            # 如果指定路径不存在且无法下载，回退到默认名称下载到缓存
            _ = YOLO(YOLO_MODEL_NAME)
    return model_path


def load_yolov8(local_path: Path) -> YOLO:
    # 优先加载指定路径，否则加载默认
    target = str(local_path) if local_path.exists() else YOLO_MODEL_NAME
    mdl = YOLO(target)
    try:
        mdl.to(YOLO_DEVICE)
    except Exception:
        pass
    return mdl


def _blip_files_ok(dir_: Path) -> bool:
    if not dir_.exists(): return False
    must_have = ["config.json", "tokenizer.json"]
    return all((dir_ / f).exists() for f in must_have)


def ensure_local_blip(local_dir: Path, repo_id: str) -> Path:
    if _blip_files_ok(local_dir):
        return local_dir
    logger.info(f"[BLIP] 本地缺失，下载 {repo_id} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    return local_dir


def load_blip(local_dir: Path) -> Optional[Dict[str, Any]]:
    if not _blip_files_ok(local_dir):
        return None
    try:
        processor = BlipProcessor.from_pretrained(str(local_dir), local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained(str(local_dir), local_files_only=True)
        model.to(BLIP_DEVICE)
        model.eval()
        return {"processor": processor, "model": model}
    except Exception as e:
        logger.error(f"[BLIP] Load error: {e}")
        return None


def _clip_files_ok(dir_: Path) -> bool:
    return dir_.exists() and (dir_ / "config.json").exists()


def ensure_local_clip(local_dir: Path, repo_id: str) -> Path:
    if _clip_files_ok(local_dir):
        return local_dir
    logger.info(f"[CLIP] 本地缺失，下载 {repo_id} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    return local_dir


def load_clip_classifier():
    global _CLF_PIPE
    if _CLF_PIPE is not None:
        return _CLF_PIPE

    local_dir = ensure_local_clip(CLIP_LOCAL_DIR, CLIP_REPO_ID)
    try:
        # 尝试加载 CLIP pipeline
        _CLF_PIPE = pipeline(
            "zero-shot-image-classification",
            model=str(local_dir),
            device=CLF_DEVICE,
        )
    except Exception as e:
        logger.error(f"[CLIP] Load error: {e}")
        _CLF_PIPE = None
    return _CLF_PIPE


# =========================
# ===== 业务逻辑函数 =====
# =========================

def fmt_score(x: float) -> str:
    return f"{x:.4f}"


def _collect_ultra(results) -> List[Dict[str, Any]]:
    dets = []
    for r in results:
        if getattr(r, "boxes", None) is None: continue
        for box in r.boxes:
            score = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            name = r.names.get(cls_id, str(cls_id))
            dets.append({"name": name, "score": score})
    return sorted(dets, key=lambda d: d["score"], reverse=True)


def detect_elements_yolo(img: Image.Image) -> List[Dict[str, Any]]:
    if not yolo_mdl: return []
    results = yolo_mdl.predict(source=img, conf=YOLO_CONF, iou=YOLO_IOU, device=YOLO_DEVICE, imgsz=YOLO_IMGSZ,
                               verbose=False, max_det=300)
    raw_dets = _collect_ultra(results)

    # 去重保留最高分
    best = {}
    for d in raw_dets:
        nm = d["name"]
        if nm not in best or d["score"] > best[nm]["score"]:
            best[nm] = d
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


@torch.no_grad()
def generate_caption_blip(image: Image.Image) -> str:
    if not blip_bundle: return ""
    proc = blip_bundle["processor"]
    model = blip_bundle["model"]

    inputs = proc(image, text="a photo of" if BLIP_USE_PROMPT else None, return_tensors="pt").to(BLIP_DEVICE)
    out = model.generate(**inputs, max_new_tokens=BLIP_MAX_TOKENS, min_new_tokens=5)
    return proc.decode(out[0], skip_special_tokens=True).strip()


def classify_scene_clip(img: Image.Image) -> List[Dict[str, Any]]:
    clf = load_clip_classifier()
    if not clf: return []

    try:
        res = clf(img, candidate_labels=CLF_LABELS_EN)
        # res 是 list[dict]
        # 找到最高分
        best_score = 0.0
        best_label = ""
        for item in res:
            if item["score"] > best_score:
                best_score = item["score"]
                best_label = item["label"]

        if best_score < 0.4: return []

        return [{
            "label_en": best_label,
            "label_zh": CLF_LABEL_ZH_MAP.get(best_label, best_label),
            "score": best_score
        }]
    except Exception as e:
        logger.error(f"[CLIP] Inference error: {e}")
        return []


# ... (辅助文本处理函数: strip_think, translate_to_zh, extract_news_entities 等) ...
# 为节省篇幅，直接复用之前脚本里的逻辑，这里简化实现

def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def translate_to_zh(llm_obj, text: str) -> str:
    if not text or not llm_obj: return text
    try:
        prompt = PromptTemplate.from_template("将以下英文翻译成中文，不带解释：\n{text}")
        chain = prompt | llm_obj | StrOutputParser() | RunnableLambda(strip_think)
        return chain.invoke({"text": text}).strip()
    except:
        return text


def fuse_zh_multi(llm_obj, scenes, targets, blips, news_text):
    if not llm_obj:
        return f"场景:{scenes}; 目标:{targets}; 描述:{blips}"

    prompt = PromptTemplate.from_template(
        """基于以下信息生成一段中文图片描述：
        新闻文本: {news}
        场景分类: {scenes}
        检测目标: {targets}
        基础描述: {blips}

        要求：自然连贯，以视觉信息为主。
        输出格式：
        label: [描述内容]
        """
    )
    chain = prompt | llm_obj | StrOutputParser() | RunnableLambda(strip_think)
    try:
        res = chain.invoke({
            "news": news_text[:200], "scenes": str(scenes),
            "targets": str(targets), "blips": str(blips)
        })
        if "label:" in res:
            return res.split("label:", 1)[1].strip()
        return res
    except:
        return "生成失败"


# =========================
# ===== 核心分析逻辑 =====
# =========================

def analyze_images_group(text: str, images_bytes: List[bytes]) -> dict:
    scenes_all = []
    targets_all = []
    blips_all = []

    for b in images_bytes:
        try:
            img = Image.open(BytesIO(b)).convert("RGB")

            # 1. 场景
            s_list = classify_scene_clip(img)
            s_str = s_list[0]["label_zh"] if s_list else "未知"
            scenes_all.append(s_str)

            # 2. 目标
            dets = detect_elements_yolo(img)
            t_str = ",".join([d['name'] for d in dets[:5]])
            targets_all.append(t_str)

            # 3. 描述
            blip_en = generate_caption_blip(img)
            blip_zh = translate_to_zh(llm, blip_en)
            blips_all.append(blip_zh)

        except Exception as e:
            logger.error(f"Image process error: {e}")

    # 融合生成
    desc = fuse_zh_multi(llm, scenes_all, targets_all, blips_all, text)

    # 简单聚合类别
    category = Counter(scenes_all).most_common(1)[0][0] if scenes_all else "未分类"

    # 聚合元素
    all_elements = set()
    for t in targets_all:
        for x in t.split(','):
            if x.strip(): all_elements.add(x.strip())

    # 翻译元素
    elements_zh = [translate_to_zh(llm, e) for e in list(all_elements)[:10]]

    return {
        "category": category,
        "elements": elements_zh,
        "description_zh": desc
    }


# =========================
# ===== 路由与预热 =====
# =========================

@router.post("/analyze", summary="多模态图片分析")
async def images_des_analyze(
        text: str = Form(""),
        images: List[UploadFile] = File(...)
):
    if not images:
        return JSONResponse(status_code=400, content={"detail": "请上传图片"})

    imgs_bytes = []
    for f in images:
        imgs_bytes.append(await f.read())

    if not yolo_mdl and not blip_bundle:
        return JSONResponse(status_code=500, content={"detail": "模型未初始化，请等待服务启动完成"})

    # 在线程池中运行耗时操作
    # Fastapi 的 async def 中运行 CPU 密集型任务会阻塞
    # 但由于这里主要是模型推理，短期内直接调用也可以，或者使用 run_in_threadpool
    try:
        result = analyze_images_group(text, imgs_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})


async def warmup():
    """系统启动时的预热函数"""
    global yolo_mdl, blip_bundle, llm

    logger.info("🔥 [ImageDesc] 开始加载多模态模型...")

    # 1. YOLO
    try:
        ensure_local_yolo_model(YOLO_LOCAL_PATH)
        yolo_mdl = load_yolov8(YOLO_LOCAL_PATH)
    except Exception as e:
        logger.error(f"❌ YOLO 加载失败: {e}")

    # 2. BLIP
    try:
        blip_dir = ensure_local_blip(BLIP_LOCAL_DIR, BLIP_REPO_ID)
        blip_bundle = load_blip(blip_dir)
    except Exception as e:
        logger.error(f"❌ BLIP 加载失败: {e}")

    # 3. CLIP
    try:
        load_clip_classifier()
    except Exception as e:
        logger.error(f"❌ CLIP 加载失败: {e}")

    # 4. LLM
    if LLM_AVAILABLE:
        try:
            llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
            # 简单测试
            # await llm.ainvoke("ping")
            logger.info(f"✅ LLM ({LLM_MODEL}) 连接成功")
        except Exception as e:
            logger.warning(f"⚠️ LLM 连接失败: {e}")
            llm = None
    else:
        logger.warning("⚠️ 未安装 langchain_ollama，LLM 功能禁用")

    logger.info("✅ [ImageDesc] 模型加载流程结束")