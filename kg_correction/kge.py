import os
import random
import time
import warnings
import torch
from typing import List, Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel

# --- 导入 Transformers ---
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except ImportError:
    warnings.warn("Transformers/Peft未安装，LLM相关功能将不可用。", ImportWarning)
    AutoTokenizer = None
    AutoModelForCausalLM = None
    PeftModel = None

# ==========================
# 1. 配置与常量
# ==========================
# 实际部署时建议将这些路径移至 config/settings.py，这里为了“放在一起”保持硬编码
BASE_MODEL_DIR = "/mnt/data/Qwen3-8B"
ADAPTER_DIR = "/blip/new tupu/runs/kgc_qwen3_nebula/checkpoint-42"

ALLOWED_EVENT_EVENT_RELATIONS = {
    "event-link-link", "event-link-parallel", "event-link-includes",
    "event-link-overlap", "event-link-alternative",
    "event-causal-trigger", "event-causal-result", "event-causal-condition",
    "event-causal-suppress", "event-causal-successor", "event-causal-reason",
    "event-causal-dependency", "event-causal-constraint",
    "event-evolution-leadsTo", "event-evolution-cause", "event-evolution-promote",
    "event-evolution-escalate", "event-evolution-deescalate", "event-evolution-transfer",
    "event-combination-subEvent", "event-combination-stage",
    "event-combination-parallelTask",
    "RELATED_TO", "PARALLEL_WITH", "INVOLVES", "INCLUDES", "CONDUCTED_BY"
}

ALLOWED_EVENT_TARGET_RELATIONS = {
    "event-target-facility-occursAtAirport", "event-target-organization-subjectOrganization",
    "event-target-person-subjectPerson", "CONDUCTED_BY", "INVOLVES"
}

ALL_RELATIONS = ALLOWED_EVENT_EVENT_RELATIONS | ALLOWED_EVENT_TARGET_RELATIONS

# ==========================
# 2. 数据模型 (Pydantic)
# ==========================
class Triple(BaseModel):
    h: str
    r: str
    t: str

class CorrectionParams(BaseModel):
    HEAD_CAND_K: int = 3
    TAIL_CAND_K: int = 3
    REL_CAND_K: int = 2
    SCORE_THRESHOLD: float = 0.80
    REL_SCORE_THRESHOLD: float = -16.0
    MAX_CTX: int = 6

class CorrectionRequest(BaseModel):
    graph: List[Triple]
    params: CorrectionParams

class CorrectionResultItem(BaseModel):
    h: str
    r: str
    t: str
    error_type: str
    pre: str
    score: float
    insert_status: str

class CorrectionResponse(BaseModel):
    results: List[CorrectionResultItem]
    used_params: Dict[str, Any]
    elapsed_ms: int

# ==========================
# 3. 核心逻辑类 (LLM Service)
# ==========================
class GraphCorrector:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """加载 LLM 模型"""
        if not AutoTokenizer:
            print("⚠️ Transformers 库未找到，跳过模型加载")
            return

        print(">>> [KGE] 正在加载 Tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"[KGE] Tokenizer 加载失败: {e}")
            return

        print(">>> [KGE] 正在加载 Model...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_DIR,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )

            if ADAPTER_DIR and os.path.exists(ADAPTER_DIR) and PeftModel:
                self.model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, ignore_mismatched_sizes=True)
                print("✅ [KGE] LoRA Adapter 加载成功")
            else:
                self.model = base_model
                print("⚠️ [KGE] 未找到 Adapter，仅使用 Base Model")

            self.model.eval()
        except Exception as e:
            print(f"[KGE] Model 加载失败: {e}")

    def guess_entity_type(self, text: str) -> str:
        """简单的规则推断实体类型"""
        text = text.strip()
        if len(text) > 8 or any(k in text for k in ["表示", "反对", "强调", "访问"]):
            return "EVENT"
        if any(k in text for k in ["外交部", "学会", "国会", "国台办"]):
            return "ORG"
        if any(k in text for k in ["人", "王毅", "郭嘉昆"]):
            return "PER"
        return "UNK"

    def rerank_labels(self, instruction, candidates):
        """LLM 打分逻辑"""
        if not self.model or not self.tokenizer:
            return random.choice(candidates) if candidates else "", [-1.0] * len(candidates)

        tok = self.tokenizer
        eos_id = tok.eos_token_id
        pad_id = tok.pad_token_id

        prompt_ids = tok(instruction, add_special_tokens=False)["input_ids"] + [eos_id]
        candidate_ids_list = [tok(c, add_special_tokens=False)["input_ids"] for c in candidates]

        seqs, seq_lens = [], []
        for c_ids in candidate_ids_list:
            seq = prompt_ids + c_ids
            seqs.append(seq)
            seq_lens.append(len(seq))

        if not seqs:
            return "", []

        maxlen = max(seq_lens)
        input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attn_mask = torch.zeros_like(input_ids)

        for i, seq in enumerate(seqs):
            input_ids[i, -len(seq):] = torch.tensor(seq, dtype=torch.long)
            attn_mask[i, -len(seq):] = 1

        input_ids = input_ids.to(self.model.device)
        attn_mask = attn_mask.to(self.model.device)

        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn_mask)

        logp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)

        scores = []
        # 计算每个候选项的得分（取末尾 token 的概率和）
        for i, seq_len in enumerate(seq_lens):
            cand_len = len(candidate_ids_list[i])
            # 累加 candidate 部分的 log_prob
            s = sum(float(logp[i, -seq_len + t, seqs[i][t]]) for t in range(seq_len - cand_len, seq_len))
            scores.append(s)

        best_idx = max(range(len(candidates)), key=lambda z: scores[z])
        return candidates[best_idx], scores

# 全局实例，供 Main 调用
corrector = GraphCorrector()

# ==========================
# 4. API 路由定义
# ==========================
router = APIRouter()

@router.post("/correct", response_model=CorrectionResponse)
async def correct_graph(request: CorrectionRequest):
    start_time = time.time()
    params = request.params
    edges = request.graph

    # 1. 构建内存图索引 (Adjacency List)
    out_edges = {}
    in_edges = {}
    for edge in edges:
        out_edges.setdefault(edge.h, []).append((edge.h, edge.r, edge.t))
        in_edges.setdefault(edge.t, []).append((edge.h, edge.r, edge.t))

    results = []

    # 2. 遍历纠错
    for edge in edges:
        h_text, r_text, t_text = edge.h, edge.r, edge.t
        h_type = corrector.guess_entity_type(h_text)

        # 检测：不在白名单视为异常
        if r_text in ALL_RELATIONS:
            continue

        # 准备候选集
        legal_rels = list(ALLOWED_EVENT_EVENT_RELATIONS if h_type == "EVENT" else ALLOWED_EVENT_TARGET_RELATIONS)
        if r_text in legal_rels:
            legal_rels.remove(r_text)

        candidates = random.sample(legal_rels, min(len(legal_rels), params.REL_CAND_K))
        if not candidates:
            continue

        # 构建上下文
        ctx = []
        neighbors = out_edges.get(h_text, []) + in_edges.get(t_text, [])
        for nh, nr, nt in neighbors:
            if nh == h_text and nt == t_text: continue
            if len(ctx) >= params.MAX_CTX: break
            ctx.append(f"{nh} {nr} {nt}")

        instr = (
            "该三元组的关系类型是错误的，请在候选集中选择唯一正确的关系名（只输出关系名）：\n"
            f"待纠错：{h_text} {r_text} {t_text}\n"
        )
        if ctx: instr += "上下文：\n" + "\n".join(f"- {c}" for c in ctx) + "\n"
        instr += "候选关系：" + ", ".join(candidates)

        # 推理
        new_rel, scores = corrector.rerank_labels(instr, candidates)
        best_score = max(scores) if scores else -float('inf')

        # 判定
        if best_score > params.REL_SCORE_THRESHOLD:
            status = "符合阈值，成功修改并插入"
            res_r = new_rel
        else:
            status = "不符合阈值条件，不修改"
            res_r = r_text

        results.append(CorrectionResultItem(
            h=h_text,
            r=res_r,
            t=t_text,
            error_type="REL_INVALID",
            pre=r_text,
            score=round(best_score, 4),
            insert_status=status
        ))

    elapsed = int((time.time() - start_time) * 1000)
    return CorrectionResponse(results=results, used_params=params.dict(), elapsed_ms=elapsed)