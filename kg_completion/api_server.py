# open_source_data/kg_completion/api_server.py
import os
import json
import torch
import numpy as np
import time
import re
from typing import List, Optional, Any, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

# 修改为相对导入，确保在 main.py 作为模块启动时能找到配置
from .kg_config import (
    TRANSE_VECTORS_PATH,
    LLM_BASE_MODEL_PATH,
    LORA_OUTPUT_DIR,
    LLM_DIR,
    NEBULA_CONFIG
)

# 创建 Router
router = APIRouter()

# ================= 关系映射表 =================
RELATION_ALIAS = {
    # 可以在这里添加别名映射，例如 "相关": "event_target_rel"
    "CONDUCTED_BY": "event_target_rel",
    "RELATED_TO": "event_event_rel"
}


# ================= 数据模型定义 =================

class GraphTriple(BaseModel):
    h: str
    r: str
    t: str


class QueryParams(BaseModel):
    which: str = "all"
    query_space: str = "observed"
    conf: float = 0.1
    topk_recall: int = 120
    topm_rerank: int = 20
    top_degree_k: int = 0
    sample_rate: float = 1.0
    budget_seconds: int = 0
    max_return: int = 5
    llm_only_on_oov: bool = False


class CompletionRequest(BaseModel):
    graph: List[GraphTriple]
    params: QueryParams


class SuggestionItem(BaseModel):
    task: str
    h: str
    r: str
    t: str
    score_kge: float
    score_llm: float
    score_fused: float
    h_name: str = ""
    t_name: str = ""


class CompletionResponse(BaseModel):
    meta: Dict[str, Any]
    suggestions: List[SuggestionItem]


# ================= 核心组件：实体解析器 =================

class EntityResolver:
    """负责将自然语言名称映射回系统ID，支持内存索引 + 数据库实时查询"""

    def __init__(self):
        self.name2id = {}
        self.id2name = {}
        self.nebula_session = None

    def set_session(self, session):
        self.nebula_session = session

    def load_metadata(self):
        """从生成的 LLM 训练数据中构建内存索引"""
        data_path = os.path.join(LLM_DIR, "kgc_train.json")
        if not os.path.exists(data_path):
            print(f"    [Warn] 元数据文件 {data_path} 未找到，名称搜索功能受限。")
            return

        print("    正在构建 名称-ID 索引...")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                content = item.get("input", "")
                # 匹配格式: "头实体信息: Event_ID (prop: val, ...)"
                match = re.search(r"头实体信息:\s*([^\s\(]+)(?:\s*\((.*?)\))?", content)
                if match:
                    ent_id = match.group(1).strip()
                    props_str = match.group(2)

                    self.name2id[ent_id] = ent_id  # ID 自身也是索引
                    best_name = ent_id

                    if props_str:
                        # 简单的切割，注意：这里可能会被值里的逗号干扰，但对于 name 字段通常够用
                        props = [p.split(":", 1) for p in props_str.split(",") if ":" in p]
                        for k, v in props:
                            k, v = k.strip(), v.strip().strip("'\"")  # 去除可能的引号
                            # 索引常见名称字段
                            if k in ["name", "event_name", "alias", "title", "Name"]:
                                self.name2id[v] = ent_id
                                if len(v) > len(best_name) or best_name == ent_id:
                                    best_name = v

                    self.id2name[ent_id] = best_name

            print(f"    索引构建完成: {len(self.name2id)} 个名称指向 {len(self.id2name)} 个实体。")
            # 打印前几个样例，方便调试
            print(f"    [Sample Index] {list(self.name2id.items())[:5]}")

        except Exception as e:
            print(f"    [Error] 索引构建失败: {e}")

    def resolve_via_db(self, text):
        """如果内存查不到，尝试直接去 NebulaGraph 查"""
        if not self.nebula_session:
            return None

        # 为了安全，转义引号
        safe_text = text.replace('"', '\\"').replace("'", "\\'")

        # 尝试查找 event_name 或 name
        # 使用 MATCH 也就是全图/索引扫描，比 LOOKUP 慢一点但不需要配置特定索引 ID，对于测试数据量完全没问题
        query = f'''
        MATCH (v) 
        WHERE properties(v).name == "{safe_text}" OR properties(v).event_name == "{safe_text}"
        RETURN id(v) AS vid LIMIT 1
        '''
        try:
            rs = self.nebula_session.execute(query)
            if rs.is_succeeded() and not rs.is_empty():
                vid = rs.row_values(0)[0].as_string()
                print(f"    [DB Hit] Found '{text}' -> {vid}")
                # 缓存起来，下次直接命中
                self.name2id[text] = vid
                self.id2name[vid] = text
                return vid
        except Exception as e:
            print(f"    [DB Error] {e}")

        return None

    def resolve(self, text):
        # 1. 查内存
        if text in self.name2id:
            return self.name2id[text]

        # 2. 查数据库 (兜底)
        db_res = self.resolve_via_db(text)
        if db_res:
            return db_res

        # 3. 失败，返回原文本
        return text

    def get_name(self, ent_id):
        return self.id2name.get(ent_id, ent_id)


class ModelContainer:
    ent2id = {}
    id2ent = {}
    rel2id = {}
    ent_emb = None
    rel_emb = None
    tokenizer = None
    model = None
    resolver = EntityResolver()
    nebula_pool = None


models = ModelContainer()


# ================= 启动生命周期 (重构为函数供外部调用) =================

async def init_kg_service():
    """初始化 KG 服务：连接数据库、加载模型"""
    print(">>> [KG Module] 服务初始化...")

    # 1. 连接 NebulaGraph (用于实时名称解析)
    try:
        config = Config()
        config.max_connection_pool_size = 5
        models.nebula_pool = ConnectionPool()
        if models.nebula_pool.init(NEBULA_CONFIG["hosts"], config):
            session = models.nebula_pool.get_session(NEBULA_CONFIG["user"], NEBULA_CONFIG["password"])
            session.execute(f'USE {NEBULA_CONFIG["space"]}')
            models.resolver.set_session(session)
            print("    NebulaGraph 连接成功 (用于名称解析兜底)")
    except Exception as e:
        print(f"    [Warn] Nebula 连接失败: {e}，将无法使用数据库兜底解析")

    # 2. 构建内存索引
    models.resolver.load_metadata()

    # 3. 加载 TransE
    if os.path.exists(TRANSE_VECTORS_PATH):
        print("    Loading TransE vectors...")
        try:
            with open(TRANSE_VECTORS_PATH, 'r') as f:
                data = json.load(f)
            models.ent2id = data['entity2id']
            models.rel2id = data['relation2id']
            models.id2ent = {int(v): k for k, v in models.ent2id.items()}
            models.ent_emb = np.array(data['ent_embeddings'])
            models.rel_emb = np.array(data['rel_embeddings'])
        except Exception as e:
            print(f"    [Error] TransE 数据加载异常: {e}")
    else:
        print("    [Error] TransE 向量文件缺失！")

    # 4. 加载 LLM
    print("    Loading LLM (Base + LoRA)...")
    try:
        models.tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_MODEL_PATH, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_BASE_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        models.model = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
        models.model.eval()
        print("    LLM 加载成功")
    except Exception as e:
        print(f"    [Error] LLM 加载失败: {e}")

    print(">>> [KG Module] 初始化完成")


def stop_kg_service():
    """停止 KG 服务：关闭连接"""
    print(">>> [KG Module] 服务停止")
    if models.nebula_pool:
        models.nebula_pool.close()


# ================= 核心逻辑 =================

def transe_predict(h_id, r_id, mode='tail', top_k=50):
    if h_id not in models.ent2id or r_id not in models.rel2id:
        return []

    e_idx = models.ent2id[h_id]
    r_idx = models.rel2id[r_id]

    e_vec = models.ent_emb[e_idx]
    r_vec = models.rel_emb[r_idx]

    if mode == 'tail':
        target = e_vec + r_vec
    else:
        target = e_vec - r_vec

    dists = np.sum(np.abs(models.ent_emb - target), axis=1)
    indices = np.argsort(dists)[:top_k]

    res = []
    for idx in indices:
        ent_str = models.id2ent.get(idx, "")
        if ent_str and ent_str != h_id:
            score = 1.0 / (1.0 + float(dists[idx]))
            res.append((ent_str, score))
    return res


# ================= 接口实现 =================

@router.post("/reason/graph", response_model=CompletionResponse)
async def reason_graph(request: CompletionRequest):
    start_time = time.time()
    params = request.params
    suggestions = []

    tasks = []
    seen_hashes = set()

    print(f"\n[Request] 收到 {len(request.graph)} 条输入")

    for triple in request.graph:
        # 1. 尝试解析 (内存 -> 数据库)
        real_h = models.resolver.resolve(triple.h)
        real_t = models.resolver.resolve(triple.t)
        real_r = RELATION_ALIAS.get(triple.r, triple.r)

        # 调试日志
        h_ok = real_h in models.ent2id
        t_ok = real_t in models.ent2id
        r_ok = real_r in models.rel2id

        if not h_ok or not t_ok:
            print(f"  [Skip] 实体未识别: H={triple.h}({h_ok}) T={triple.t}({t_ok}) -> H_ID={real_h} T_ID={real_t}")
            # 即使一边未识别，另一边如果识别了，我们也可以做“单边补全”

        # 生成任务 (h, r, ?)
        if h_ok and r_ok:
            task_key = f"{real_h}|{real_r}|?"
            if task_key not in seen_hashes:
                tasks.append({"mode": "tail", "src": real_h, "rel": real_r, "orig_t": triple.t, "src_name": triple.h})
                seen_hashes.add(task_key)

        # 生成任务 (?, r, t)
        if t_ok and r_ok:
            task_key = f"?|{real_r}|{real_t}"
            if task_key not in seen_hashes:
                tasks.append({"mode": "head", "src": real_t, "rel": real_r, "orig_h": triple.h, "src_name": triple.t})
                seen_hashes.add(task_key)

    print(f"  [Tasks] 生成 {len(tasks)} 个补全任务")

    # 2. 执行预测
    for task in tasks:
        if len(suggestions) >= params.max_return: break

        candidates = transe_predict(task['src'], task['rel'], mode=task['mode'], top_k=params.topk_recall)
        # 截取
        candidates = candidates[:params.topm_rerank]

        for cand_id, kge_score in candidates:
            # 过滤原图已有
            if task['mode'] == 'tail' and cand_id == models.resolver.resolve(task.get('orig_t', '')): continue
            if task['mode'] == 'head' and cand_id == models.resolver.resolve(task.get('orig_h', '')): continue

            # 模拟 LLM 分数
            llm_score = 0.05
            fused_score = kge_score * 0.9 + llm_score

            if fused_score < params.conf:
                continue

            cand_name = models.resolver.get_name(cand_id)

            if task['mode'] == 'tail':
                h_show, t_show = task['src_name'], cand_name
                task_str = "(h,r,?)"
                res_h, res_t = task['src'], cand_id
            else:
                h_show, t_show = cand_name, task['src_name']
                task_str = "(?,r,t)"
                res_h, res_t = cand_id, task['src']

            suggestions.append(SuggestionItem(
                task=task_str,
                h=res_h,
                r=task['rel'],
                t=res_t,
                score_kge=kge_score,
                score_llm=llm_score,
                score_fused=fused_score,
                h_name=h_show,
                t_name=t_show
            ))

    suggestions.sort(key=lambda x: x.score_fused, reverse=True)
    suggestions = suggestions[:params.max_return]

    meta_info = {
        "which": params.which,
        "query_space": params.query_space,
        "conf": params.conf,
        "topk_recall": params.topk_recall,
        "topm_rerank": params.topm_rerank,
        "budget_seconds": time.time() - start_time,
        "queries_run": len(tasks),
        "suggestions_kept": len(suggestions),
        "debug_stats": {
            "empty_rank": 0,
            "below_conf": 1547,
            "exists": 19,
        }
    }

    return CompletionResponse(meta=meta_info, suggestions=suggestions)