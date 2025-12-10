import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

# ================= 配置区域 =================
# 1. 路径配置
PROJECT_ROOT = "/open_source_data/kg_completion"
TRANSE_VEC_PATH = os.path.join(PROJECT_ROOT, "transe_vectors.json")
TRANSE_DATA_DIR = os.path.join(PROJECT_ROOT, "kgc_data/transe")
LLM_BASE_PATH = "/mnt/data/Qwen3-8B"
LLM_LORA_PATH = "/open_source_data/checkpoints/qwen_kgc_lora"

# 2. Nebula 连接配置
NEBULA_IP = "39.104.200.88"
NEBULA_PORT = 41003
USER = "root"
PASSWORD = "123456"
SPACE_NAME = "event_target1"

# 3. 推理参数
TOP_K_CANDIDATES = 5  # TransE 推荐前几个给 LLM 选
CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值 (可选)


# ================= 核心类定义 =================

class NeuroSymbolicReasoner:
    def __init__(self):
        print(">>> [1/4] 正在初始化模型与连接...")
        self._init_nebula()
        self._init_transe()
        self._init_llm()
        print(">>> 初始化完成。")

    def _init_nebula(self):
        config = Config()
        config.max_connection_pool_size = 10
        self.pool = ConnectionPool()
        if not self.pool.init([(NEBULA_IP, NEBULA_PORT)], config):
            raise RuntimeError("无法连接到 NebulaGraph")
        self.session = self.pool.get_session(USER, PASSWORD)
        self.session.execute(f'USE {SPACE_NAME}')

    def _init_transe(self):
        # 1. 加载 ID 映射
        self.ent2id = {}
        self.id2ent = {}
        self.rel2id = {}

        with open(os.path.join(TRANSE_DATA_DIR, 'entity2id.txt'), 'r') as f:
            f.readline()  # 跳过计数行
            for line in f:
                ent, idx = line.strip().split('\t')
                self.ent2id[ent] = int(idx)
                self.id2ent[int(idx)] = ent

        with open(os.path.join(TRANSE_DATA_DIR, 'relation2id.txt'), 'r') as f:
            f.readline()
            for line in f:
                rel, idx = line.strip().split('\t')
                self.rel2id[rel] = int(idx)

        # 2. 加载向量
        with open(TRANSE_VEC_PATH, 'r') as f:
            vec_data = json.load(f)
            self.ent_embeddings = np.array(vec_data['ent_embeddings'])
            self.rel_embeddings = np.array(vec_data['rel_embeddings'])

    def _init_llm(self):
        print("    正在加载 LLM (这也需要一点时间)...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_PATH, trust_remote_code=True)
        # 加载基座
        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_BASE_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        # 加载 LoRA
        self.model = PeftModel.from_pretrained(base_model, LLM_LORA_PATH)
        self.model.eval()

    def get_node_properties(self, entity_id):
        """从 Nebula 获取节点的详细属性文本"""
        # 简单处理：假设 ID 是字符串，需要加引号
        query = f'MATCH (v) WHERE id(v) == "{entity_id}" RETURN properties(v)'
        result = self.session.execute(query)
        if result.is_succeeded() and not result.is_empty():
            props = result.row_values(0)[0].as_map()
            # 格式化为字符串
            desc_parts = [f"{k}: {v}" for k, v in props.items() if v]
            return f"{entity_id} [{'; '.join(desc_parts)}]"
        return str(entity_id)

    def transe_recall(self, head_id, relation_name):
        """TransE 阶段：根据 h + r 计算最接近的 TOP-K 尾实体"""
        if head_id not in self.ent2id or relation_name not in self.rel2id:
            print(f"警告: 实体 {head_id} 或关系 {relation_name} 不在 TransE 词表中，跳过结构召回。")
            return []

        h_idx = self.ent2id[head_id]
        r_idx = self.rel2id[relation_name]

        h_vec = self.ent_embeddings[h_idx]
        r_vec = self.rel_embeddings[r_idx]

        # TransE 原理: h + r ≈ t  =>  min |(h+r) - t|
        target_vec = h_vec + r_vec

        # 计算与所有实体的距离 (使用 L1 距离，广播计算)
        distances = np.sum(np.abs(self.ent_embeddings - target_vec), axis=1)

        # 获取距离最小的前 K 个索引
        top_k_indices = np.argsort(distances)[:TOP_K_CANDIDATES]

        candidates = []
        for idx in top_k_indices:
            ent_id = self.id2ent[idx]
            # 排除掉头实体自己（如果是自环则保留，通常补全不补自己）
            if ent_id != head_id:
                candidates.append(ent_id)

        return candidates

    def llm_reasoning(self, head_desc, relation_name, candidate_ids):
        """LLM 阶段：从候选者中选出最佳答案"""
        if not candidate_ids:
            return None

        # 1. 获取候选者的详细描述，辅助 LLM 判断
        candidates_text = []
        for cid in candidate_ids:
            # 这里为了速度，只取 ID，如果需要更精准，可以查 Nebula 取属性
            candidates_text.append(f"- {cid}")

        candidates_str = "\n".join(candidates_text)

        # 2. 构造 Prompt
        prompt = (
            f"任务：知识图谱补全。请根据图结构推荐的候选列表，结合语义逻辑，选出最正确的尾实体。\n\n"
            f"头实体信息：{head_desc}\n"
            f"待补全关系：{relation_name}\n\n"
            f"TransE推荐的候选实体：\n{candidates_str}\n\n"
            f"请直接输出正确的实体ID（必须在候选列表中），不要输出其他废话。"
        )

        # 3. 生成
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_ids,
                max_new_tokens=64,  # ID 通常不长
                temperature=0.1  # 低温采样，保证确定性
            )

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # 提取 LLM 回复的最后一部分（去掉 Prompt）
        response_clean = response.split(text)[-1].strip()

        # 简单的后处理：检查 LLM 输出是否在候选列表中
        # 如果 LLM 输出了 "是 Entity_A"，我们需要提取 "Entity_A"
        for cid in candidate_ids:
            if cid in response_clean:
                return cid

        return None  # 如果没匹配上，保守起见不补全

    def write_back(self, head_id, relation, tail_id):
        """将预测结果写入 NebulaGraph"""
        # 注意：这里我们添加一个属性 source='AI_Reasoning' 以示区分
        # 假设边的类型就是 relation 字符串
        edge_ql = f'INSERT EDGE `{relation}` (relation, source) VALUES "{head_id}"->"{tail_id}":("{relation}", "AI_Reasoning");'

        try:
            resp = self.session.execute(edge_ql)
            if resp.is_succeeded():
                print(f"  [成功回写] {head_id} --[{relation}]--> {tail_id}")
            else:
                print(f"  [回写失败] {resp.error_msg()}")
        except Exception as e:
            print(f"  [回写异常] {e}")

    def predict_and_complete(self, head_id, relation_type):
        print(f"\n--- 处理任务: ({head_id}, {relation_type}, ?) ---")

        # 1. TransE 召回
        candidates = self.transe_recall(head_id, relation_type)
        print(f"  TransE 推荐: {candidates}")

        if not candidates:
            return

        # 2. 获取头实体语义
        head_desc = self.get_node_properties(head_id)

        # 3. LLM 决策
        final_tail = self.llm_reasoning(head_desc, relation_type, candidates)

        if final_tail:
            print(f"  LLM 最终决策: {final_tail}")
            # 4. 回写数据库
            self.write_back(head_id, relation_type, final_tail)
        else:
            print("  LLM 未能做出明确选择，跳过。")

    def close(self):
        self.session.release()
        self.pool.close()


# ================= 主程序入口 =================

if __name__ == "__main__":
    reasoner = NeuroSymbolicReasoner()

    try:
        # 示例：手动指定几个需要补全的任务进行测试
        # 在实际生产中，这里可以遍历数据库中度数较少的节点

        test_cases = [
            # (头实体ID, 关系类型) -> 请替换为您数据库中真实存在的测试数据
            ("Th_MIL-00001", "event-causal-result"),
            ("Th_POL-00044", "event-target-organization-subjectOrganization"),
            ("Th_ECON-00030", "event-target-person-subjectPerson")
        ]

        for h, r in test_cases:
            reasoner.predict_and_complete(h, r)

    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        reasoner.close()