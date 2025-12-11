# 文件路径: open_source_data/kg_completion/step4_inference.py
import json
import torch
import numpy as np
import random
import os
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from kg_config import NEBULA_CONFIG, TRANSE_VECTORS_PATH, LLM_BASE_MODEL_PATH, LORA_OUTPUT_DIR, TRANSE_DIR


class NeuroSymbolicReasoner:
    def __init__(self):
        print(">>> [Step 4] 初始化推理引擎...")

        # 1. 连接 Nebula
        config = Config()
        config.max_connection_pool_size = 5
        self.pool = ConnectionPool()
        if not self.pool.init(NEBULA_CONFIG["hosts"], config):
            raise RuntimeError("Nebula 连接失败")
        self.session = self.pool.get_session(NEBULA_CONFIG["user"], NEBULA_CONFIG["password"])
        self.session.execute(f'USE {NEBULA_CONFIG["space"]}')

        # 2. 加载 TransE 数据
        print("    加载 TransE 向量...")
        with open(TRANSE_VECTORS_PATH, 'r') as f:
            data = json.load(f)
        self.ent2id = data['entity2id']
        self.rel2id = data['relation2id']
        self.id2ent = {int(v): k for k, v in self.ent2id.items()}
        self.ent_emb = np.array(data['ent_embeddings'])
        self.rel_emb = np.array(data['rel_embeddings'])

        # 加载 关系映射表 (用于回写)
        self.relation_map = {}
        map_path = os.path.join(TRANSE_DIR, 'relation_map.json')
        if os.path.exists(map_path):
            with open(map_path, 'r') as f:
                self.relation_map = json.load(f)

        # 3. 加载 LLM
        print("    加载 LLM (Base + LoRA)...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_MODEL_PATH, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_BASE_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
        self.model.eval()

    def close(self):
        if self.session: self.session.release()
        if self.pool: self.pool.close()

    def transe_recall(self, h_id, r_name, top_k=10):
        if h_id not in self.ent2id or r_name not in self.rel2id:
            print(f"    [Warn] '{h_id}' 或 '{r_name}' 不在词表中")
            return []

        h_idx = self.ent2id[h_id]
        r_idx = self.rel2id[r_name]
        target = self.ent_emb[h_idx] + self.rel_emb[r_idx]
        dists = np.sum(np.abs(self.ent_emb - target), axis=1)
        indices = np.argsort(dists)[:top_k]

        return [self.id2ent[i] for i in indices if self.id2ent[i] != h_id]

    def llm_reason(self, h_desc, r_name, candidates):
        if not candidates: return None
        cands_str = "\n".join([f"- {c}" for c in candidates])
        prompt = (f"任务：知识图谱补全。\n头实体：{h_desc}\n关系：{r_name}\n"
                  f"候选：\n{cands_str}\n请选择正确的实体ID：")

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(inputs.input_ids, max_new_tokens=64, temperature=0.1)
        resp = self.tokenizer.decode(output[0], skip_special_tokens=True)

        for cand in candidates:
            if cand in resp: return cand
        return None

    def execute_task(self, head_id, relation_val, true_tail=None):
        print(f"\n--- 任务: {head_id} --[{relation_val}]--> ? ---")

        candidates = self.transe_recall(head_id, relation_val)
        if not candidates: return

        # 获取属性
        safe_id = head_id.replace('"', '\\"')
        rs = self.session.execute(f'MATCH (v) WHERE id(v) == "{safe_id}" RETURN properties(v)')
        h_desc = head_id
        if rs.is_succeeded() and not rs.is_empty():
            props = rs.row_values(0)[0].as_map()
            desc = [f"{k}:{v}" for k, v in props.items() if v]
            h_desc = f"{head_id} ({','.join(desc)})"

        final = self.llm_reason(h_desc, relation_val, candidates)

        if final:
            print(f"  LLM 决策: {final}")
            # === 核心修正：回写逻辑 ===
            # 1. 查找对应的 Edge Type (如 event_event_rel)
            edge_type = self.relation_map.get(relation_val)
            if not edge_type:
                print(f"  [Error] 无法找到关系 '{relation_val}' 对应的 Edge Type，无法回写。")
                return

            # 2. 构造插入语句，将 relation_val 作为属性值写入
            # INSERT EDGE event_event_rel (relation, source) VALUES "h"->"t":("event-causal-result", "AI")
            gql = f'INSERT EDGE `{edge_type}` (relation, source) VALUES "{head_id}"->"{final}":("{relation_val}", "AI_Reasoning");'

            resp = self.session.execute(gql)
            if resp.is_succeeded():
                print("  ✅ 已成功回写到 NebulaGraph")
            else:
                print(f"  ❌ 回写失败: {resp.error_msg()}")


if __name__ == "__main__":
    bot = NeuroSymbolicReasoner()
    try:
        # 测试逻辑...
        pass
    finally:
        bot.close()