import json
import torch
import numpy as np
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # 关键库：用于加载 LoRA
from kg_config import NEBULA_CONFIG, TRANSE_VECTORS_PATH, LLM_BASE_MODEL_PATH, LORA_OUTPUT_DIR


class NeuroSymbolicReasoner:
    def __init__(self):
        print(">>> [Step 4] 初始化推理引擎...")

        # 1. 连接 NebulaGraph
        config = Config();
        config.max_connection_pool_size = 5
        self.pool = ConnectionPool()
        if not self.pool.init(NEBULA_CONFIG["hosts"], config):
            raise RuntimeError("Nebula 连接失败")
        self.session = self.pool.get_session(NEBULA_CONFIG["user"], NEBULA_CONFIG["password"])
        self.session.execute(f'USE {NEBULA_CONFIG["space"]}')

        # 2. 加载 TransE 向量
        print("    加载 TransE 向量...")
        with open(TRANSE_VECTORS_PATH, 'r') as f:
            data = json.load(f)
        self.ent2id = data['entity2id']
        self.rel2id = data['relation2id']
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.ent_emb = np.array(data['ent_embeddings'])
        self.rel_emb = np.array(data['rel_embeddings'])

        # 3. 加载 LLM (Base + LoRA)
        print("    加载 LLM (Base + LoRA)...")
        # A. 加载底座
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_MODEL_PATH, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_BASE_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # 显存优化
            trust_remote_code=True
        )

        # B. 挂载 LoRA 权重 (不进行 Merge，节省内存和加载时间)
        self.model = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
        self.model.eval()
        print("    模型加载完成。")

    def transe_recall(self, h_id, r_name, top_k=10):
        """TransE 粗排召回"""
        if h_id not in self.ent2id or r_name not in self.rel2id:
            return []

        h_idx = self.ent2id[h_id]
        r_idx = self.rel2id[r_name]
        target_vec = self.ent_emb[h_idx] + self.rel_emb[r_idx]  # h + r ≈ t

        # 计算距离
        dists = np.sum(np.abs(self.ent_emb - target_vec), axis=1)
        indices = np.argsort(dists)[:top_k]

        candidates = []
        for idx in indices:
            ent = self.id2ent[idx]
            if ent != h_id: candidates.append(ent)
        return candidates

    def llm_reason(self, h_desc, r_name, candidates):
        """LLM 精排推理"""
        if not candidates: return None

        cands_str = "\n".join([f"- {c}" for c in candidates])
        prompt = (
            f"任务：从候选列表中选出最可能的尾实体。\n"
            f"头实体：{h_desc}\n"
            f"关系：{r_name}\n"
            f"候选列表：\n{cands_str}\n"
            f"请只输出正确的实体名称/ID，不要输出其他内容。"
        )

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(inputs.input_ids, max_new_tokens=64, temperature=0.1)

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.split(text)[-1].strip()  # 截取生成部分

        # 简单验证
        for cand in candidates:
            if cand in response:
                return cand
        return None

    def execute_task(self, head_id, relation_type):
        print(f"\n--- 任务: {head_id} --[{relation_type}]--> ? ---")

        # 1. 结构召回
        candidates = self.transe_recall(head_id, relation_type)
        print(f"  TransE 召回: {candidates}")
        if not candidates: return

        # 2. 语义推理
        # 获取头实体属性
        props_query = f'MATCH (v) WHERE id(v) == "{head_id}" RETURN properties(v)'
        rs = self.session.execute(props_query)
        h_desc = head_id
        if rs.is_succeeded() and not rs.is_empty():
            props = rs.row_values(0)[0].as_map()
            desc_list = [f"{k}:{v}" for k, v in props.items() if v]
            h_desc = f"{head_id} ({','.join(desc_list)})"

        final_result = self.llm_reason(h_desc, relation_type, candidates)

        # 3. 回写数据库
        if final_result:
            print(f"  LLM 判定结果: {final_result}")
            # 使用反引号包裹 edge type，防止特殊字符报错
            gql = f'INSERT EDGE `{relation_type}` (source) VALUES "{head_id}"->"{final_result}":("AI_Reasoning");'
            self.session.execute(gql)
            print("  已回写至 Nebula。")
        else:
            print("  LLM 无法确定，跳过。")


if __name__ == "__main__":
    bot = NeuroSymbolicReasoner()

    # 示例测试
    test_cases = [
        ("Event_001", "event-causal-result"),
        ("Person_Zhang", "person-works-for")
    ]

    for h, r in test_cases:
        try:
            bot.execute_task(h, r)
        except Exception as e:
            print(f"任务出错: {e}")