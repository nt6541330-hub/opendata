import json
import os
import numpy as np
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from kg_config import NEBULA_CONFIG, TRANSE_DIR, LLM_DIR


def format_props(entity_id, props_map):
    """将属性 Map 转换为自然语言描述"""
    if not props_map: return str(entity_id)
    # 过滤掉无意义字段
    desc = [f"{k}: {v}" for k, v in props_map.items() if v and k not in ['vid', 'tag', 'vertex_id']]
    return f"{entity_id} ({', '.join(desc)})" if desc else str(entity_id)


def main():
    print(">>> [Step 1] 连接 NebulaGraph 并导出数据...")

    config = Config()
    config.max_connection_pool_size = 10
    pool = ConnectionPool()
    if not pool.init(NEBULA_CONFIG["hosts"], config):
        raise RuntimeError("连接 Nebula 失败")

    raw_data = []
    entities = set()
    relations = set()

    with pool.session_context(NEBULA_CONFIG["user"], NEBULA_CONFIG["password"]) as session:
        session.execute(f'USE {NEBULA_CONFIG["space"]}')

        # 修正: 使用 type(e) 获取真实的 Edge Type
        query = """
        MATCH (h)-[e]->(t) 
        RETURN 
            id(h) AS h_id, properties(h) AS h_props,
            type(e) AS rel_type, 
            id(t) AS t_id, properties(t) AS t_props
        LIMIT 20000;
        """
        print("    正在执行查询 (Limit 20000)...")
        result = session.execute(query)
        if not result.is_succeeded():
            raise RuntimeError(f"查询失败: {result.error_msg()}")

        print(f"    获取到 {result.row_size()} 条数据。")

        for i in range(result.row_size()):
            row = result.row_values(i)
            h_id = row[0].as_string()
            r_type = row[2].as_string()
            t_id = row[3].as_string()

            h_props = row[1].as_map() if row[1].is_map() else {}
            t_props = row[4].as_map() if row[4].is_map() else {}

            # 清洗双引号，避免 JSON 报错
            h_props = {k: str(v).replace('"', "'") for k, v in h_props.items()}
            t_props = {k: str(v).replace('"', "'") for k, v in t_props.items()}

            raw_data.append({
                "h": h_id, "r": r_type, "t": t_id,
                "h_desc": format_props(h_id, h_props),
                "t_desc": format_props(t_id, t_props)
            })
            entities.add(h_id)
            entities.add(t_id)
            relations.add(r_type)

    # --- 1. 生成 TransE 训练数据 ---
    print("    生成 TransE 数据...")
    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}

    # 保存映射 (后续推理需要)
    with open(os.path.join(TRANSE_DIR, 'entity2id.json'), 'w') as f:
        json.dump(ent2id, f)
    with open(os.path.join(TRANSE_DIR, 'relation2id.json'), 'w') as f:
        json.dump(rel2id, f)

    # 保存训练三元组
    triples = []
    for item in raw_data:
        triples.append((ent2id[item['h']], rel2id[item['r']], ent2id[item['t']]))

    np.save(os.path.join(TRANSE_DIR, 'train_triples.npy'), np.array(triples))

    # --- 2. 生成 LLM 微调数据 ---
    print("    生成 LLM 微调数据...")
    sft_data = []
    for item in raw_data:
        sft_data.append({
            "instruction": "知识图谱补全任务：根据头实体和关系类型，预测尾实体。",
            "input": f"头实体信息: {item['h_desc']}\n关系类型: {item['r']}",
            "output": item['t_desc']  # 让模型学会输出包含属性的完整描述
        })

    with open(os.path.join(LLM_DIR, 'kgc_train.json'), 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(">>> [Step 1] 数据导出完成。")


if __name__ == "__main__":
    main()