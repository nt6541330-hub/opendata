import os
import json
import random
import time
import warnings
import torch
import numpy as np
from typing import List, Dict, Optional, Set, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- 导入 Transformers (原脚本逻辑) ---
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except ImportError:
    warnings.warn("Transformers/Peft未安装，LLM相关功能将不可用。", ImportWarning)
    AutoTokenizer = None
    AutoModelForCausalLM = None
    PeftModel = None

# ==========================
# 1. 配置与常量 (复用原脚本)
# ==========================

# 模拟配置，实际部署时请修改为真实路径
BASE_MODEL_DIR = "/mnt/data/Qwen3-8B"
ADAPTER_DIR = "/blip/new tupu/runs/kgc_qwen3_nebula/checkpoint-42"
NONE_TOKEN = "<NONE>"

# 关系白名单 (复用原脚本)
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
    # 兼容用户输入的简化关系名
    "RELATED_TO", "PARALLEL_WITH", "INVOLVES", "INCLUDES", "CONDUCTED_BY"
}

ALLOWED_EVENT_TARGET_RELATIONS = {
    "event-target-facility-occursAtAirport", "event-target-organization-subjectOrganization",
    "event-target-person-subjectPerson", "CONDUCTED_BY", "INVOLVES"
}

ALL_RELATIONS = ALLOWED_EVENT_EVENT_RELATIONS | ALLOWED_EVENT_TARGET_RELATIONS


# ==========================
# 2. 核心逻辑类 (LLM & Logic)
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

        print("正在加载 Tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Tokenizer 加载失败: {e}")
            return

        print("正在加载 Model...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_DIR,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )

            if ADAPTER_DIR and os.path.exists(ADAPTER_DIR) and PeftModel:
                self.model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, ignore_mismatched_sizes=True)
                print("✅ LoRA Adapter 加载成功")
            else:
                self.model = base_model
                print("⚠️ 未找到 Adapter，仅使用 Base Model")

            self.model.eval()
        except Exception as e:
            print(f"Model 加载失败: {e}")

    def guess_entity_type(self, text: str) -> str:
        """
        [新增] 简单的规则推断实体类型，适配用户输入的中文文本
        原脚本依赖 'TH-', 'ORG-' 前缀，这里适配为文本特征。
        """
        text = text.strip()
        if len(text) > 8 or "表示" in text or "反对" in text or "强调" in text or "访问" in text:
            return "EVENT"
        if "外交部" in text or "学会" in text or "国会" in text or "国台办" in text:
            return "ORG"
        if "人" in text or "王毅" in text or "郭嘉昆" in text:
            return "PER"
        return "UNK"

    def get_legal_relations(self, src_type, dst_type):
        """根据实体类型返回合法关系候选集"""
        # 这里简化处理，直接返回所有关系用于演示。
        # 实际逻辑应复用原脚本 check_event_event 等函数的反向映射
        return list(ALL_RELATIONS)

    def rerank_labels(self, instruction, candidates):
        """复用原脚本的 LLM 打分逻辑"""
        if not self.model or not self.tokenizer:
            # 如果没有模型，返回随机结果用于测试 API 通通
            return random.choice(candidates), [-1.0] * len(candidates)

        tok = self.tokenizer
        eos_id = tok.eos_token_id
        pad_id = tok.pad_token_id

        prompt_ids = tok(instruction, add_special_tokens=False)["input_ids"] + [eos_id]
        candidate_ids_list = [tok(c, add_special_tokens=False)["input_ids"] for c in candidates]

        seqs, seq_lens, lab_lens = [], [], []
        for c_ids in candidate_ids_list:
            seq = prompt_ids + c_ids
            seqs.append(seq)
            seq_lens.append(len(seq))
            lab_lens.append(len(c_ids))

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
        for i, (L, ll) in enumerate(zip(seq_lens, lab_lens)):
            s = sum(float(logp[i, -L + t, seqs[i][t]]) for t in range(L - ll, L))
            scores.append(s)

        best_idx = max(range(len(candidates)), key=lambda z: scores[z])
        return candidates[best_idx], scores


# 全局实例
corrector = GraphCorrector()


# ==========================
# 3. Pydantic Models (API 契约)
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
# 4. FastAPI App & Logic
# ==========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    corrector.load_model()
    yield
    # 关闭时清理 (可选)
    pass


app = FastAPI(lifespan=lifespan)


@app.post("/api/kg/correct", response_model=CorrectionResponse)
async def correct_graph(request: CorrectionRequest):
    start_time = time.time()
    params = request.params
    edges = request.graph

    # 1. 构建内存图索引 (Adjacency List) 用于生成 Context
    # 替代原脚本的 Nebula 查询
    out_edges = {}
    in_edges = {}

    for edge in edges:
        if edge.h not in out_edges: out_edges[edge.h] = []
        out_edges[edge.h].append((edge.h, edge.r, edge.t))

        if edge.t not in in_edges: in_edges[edge.t] = []
        in_edges[edge.t].append((edge.h, edge.r, edge.t))

    results = []

    # 2. 遍历纠错
    for edge in edges:
        h_text, r_text, t_text = edge.h, edge.r, edge.t
        h_type = corrector.guess_entity_type(h_text)
        t_type = corrector.guess_entity_type(t_text)

        # --- 检测逻辑 ---
        # 简单规则检测：如果关系不在白名单，或者类型明显不匹配
        # (此处简化了 check_edge_rules 的实现以适配通用文本)
        is_bad = False
        if r_text not in ALL_RELATIONS:
            is_bad = True

        # 模拟 TransE 检测 (因无 ID 系统，此处仅作为演示逻辑)
        # 如果需要集成 TransE，需要加载 embedding 并计算 score

        if not is_bad:
            continue  # 边是好的，跳过

        # --- 纠错逻辑 ---
        # 生成候选关系
        legal_rels = list(ALLOWED_EVENT_EVENT_RELATIONS if h_type == "EVENT" else ALLOWED_EVENT_TARGET_RELATIONS)
        if r_text in legal_rels:
            legal_rels.remove(r_text)

        candidates = random.sample(legal_rels, min(len(legal_rels), params.REL_CAND_K))
        if not candidates:
            continue

        # 构建 Prompt Context
        ctx = []
        # 从构建的内存图中获取邻居
        neighbors = out_edges.get(h_text, []) + in_edges.get(t_text, [])
        for nh, nr, nt in neighbors:
            if nh == h_text and nt == t_text: continue  # 跳过自己
            if len(ctx) >= params.MAX_CTX: break
            ctx.append(f"{nh} {nr} {nt}")

        instr = (
            "该三元组的关系类型是错误的，请在候选集中选择唯一正确的关系名（只输出关系名）：\n"
            f"待纠错：{h_text} {r_text} {t_text}\n"
        )
        if ctx: instr += "上下文：\n" + "\n".join(f"- {c}" for c in ctx) + "\n"
        instr += "候选关系：" + ", ".join(candidates)

        # LLM 推理
        new_rel, scores = corrector.rerank_labels(instr, candidates)
        best_score = max(scores)

        # 构造结果
        # 注意：原脚本主要做关系纠错，这里仅展示关系纠错 (REL_INVALID)
        # 如果需要 HEAD_INVALID，需要另外的实体替换逻辑
        if best_score > params.REL_SCORE_THRESHOLD:
            status = "符合阈值，成功修改并插入"
            res_r = new_rel
        else:
            status = "不符合阈值条件，不修改"
            res_r = r_text  # 保持原样

        # 添加到结果集
        results.append(CorrectionResultItem(
            h=h_text,
            r=res_r,
            t=t_text,
            error_type="REL_INVALID",  # 原脚本主要支持关系纠错
            pre=r_text,  # 原始值
            score=round(best_score, 4),
            insert_status=status
        ))

    elapsed = int((time.time() - start_time) * 1000)

    return CorrectionResponse(
        results=results,
        used_params=params.dict(),
        elapsed_ms=elapsed
    )


if __name__ == "__main__":
    import uvicorn

    # 启动命令
    uvicorn.run(app, host="0.0.0.0", port=8803)