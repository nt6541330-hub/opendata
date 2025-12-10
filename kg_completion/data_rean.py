import json
import os
import time
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

# ==========================
# 1. Nebula 连接配置
# ==========================
GRAPH_IP = "39.104.200.88"
GRAPH_PORT = 41003
USER = "root"
PASSWORD = "123456"
SPACE_NAME = "event_target1"

# 输出目录配置
OUTPUT_DIR = './kgc_data'
TRANSE_DIR = os.path.join(OUTPUT_DIR, 'transe')
LLM_DIR = os.path.join(OUTPUT_DIR, 'llm')

# 创建文件夹
for d in [OUTPUT_DIR, TRANSE_DIR, LLM_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)


# ==========================
# 2. 工具函数
# ==========================

def format_props_to_text(entity_id, props_map):
    """
    将实体属性转换为文本描述
    例如: "Event_01 [name: 军演; time: 2025; location: 南海]"
    """
    if not props_map:
        return str(entity_id)

    desc_parts = []
    # 过滤掉无意义的属性，如有需要可在此添加黑名单
    ignore_keys = ['vid', 'tag']

    for k, v in props_map.items():
        if k in ignore_keys or v is None or v == "" or v == "null":
            continue
        # 处理值，如果是字符串且包含引号，简单清理一下
        val_str = str(v).replace('\n', ' ').replace('"', "'")
        desc_parts.append(f"{k}: {val_str}")

    if not desc_parts:
        return str(entity_id)

    return f"{entity_id} [{'; '.join(desc_parts)}]"


def execute_query(session, query):
    result = session.execute(query)
    if not result.is_succeeded():
        raise RuntimeError(f"查询失败: {result.error_msg()}")
    return result


# ==========================
# 3. 主逻辑
# ==========================

def export_and_process():
    print(f"正在连接 NebulaGraph ({GRAPH_IP}:{GRAPH_PORT})...")

    config = Config()
    config.max_connection_pool_size = 10
    connection_pool = ConnectionPool()

    if not connection_pool.init([(GRAPH_IP, GRAPH_PORT)], config):
        raise Exception("无法连接到 NebulaGraph，请检查 IP 和端口")

    # 用于 TransE 的 ID 映射
    entities_set = set()
    relations_set = set()

    # 暂存数据
    raw_data = []

    try:
        with connection_pool.session_context(USER, PASSWORD) as session:
            print(f"切换空间至: {SPACE_NAME}")
            execute_query(session, f'USE {SPACE_NAME}')

            # 构造查询：同时获取 ID、节点属性、边属性(relation)
            # LIMIT 10000 仅供演示，全量导出请考虑使用分页 SCAN 或去掉 LIMIT
            query = """
            MATCH (h)-[e]->(t) 
            RETURN 
                id(h) AS h_id, properties(h) AS h_props,
                properties(e).relation AS rel_type, 
                id(t) AS t_id, properties(t) AS t_props
            LIMIT 10000;
            """

            print("正在执行图查询...")
            result = execute_query(session, query)
            size = result.row_size()
            print(f"查询成功，获取到 {size} 条三元组，开始处理...")

            if size == 0:
                print("警告: 数据库中没有查到数据！")
                return

            for i in range(size):
                row = result.row_values(i)

                # 1. 解析基础数据
                h_id = row[0].as_string()
                # h_props 需要转为 dict
                h_props = {}
                if row[1] and row[1].is_map():
                    h_props_val = row[1].as_map()
                    h_props = {k: (v.as_string() if v.is_string() else str(v)) for k, v in h_props_val.items()}

                # 获取边上的细粒度关系
                r_val = row[2]
                rel_type = r_val.as_string() if (r_val and r_val.is_string()) else "unknown_relation"

                t_id = row[3].as_string()
                t_props = {}
                if row[4] and row[4].is_map():
                    t_props_val = row[4].as_map()
                    t_props = {k: (v.as_string() if v.is_string() else str(v)) for k, v in t_props_val.items()}

                # 2. 收集 ID (TransE 用)
                entities_set.add(h_id)
                entities_set.add(t_id)
                relations_set.add(rel_type)

                # 3. 生成语义描述 (LLM 用)
                h_text = format_props_to_text(h_id, h_props)
                t_text = format_props_to_text(t_id, t_props)

                raw_data.append({
                    "h_id": h_id,
                    "r_val": rel_type,
                    "t_id": t_id,
                    "h_desc": h_text,
                    "t_desc": t_text
                })

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        connection_pool.close()

    # ==========================
    # 4. 生成 TransE 数据
    # ==========================
    print("\n[1/2] 正在生成 TransE 训练数据...")

    # 排序并建立映射
    entity_list = sorted(list(entities_set))
    relation_list = sorted(list(relations_set))

    ent2id = {ent: i for i, ent in enumerate(entity_list)}
    rel2id = {rel: i for i, rel in enumerate(relation_list)}

    # 写 entity2id.txt
    with open(os.path.join(TRANSE_DIR, 'entity2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(entity_list)}\n")
        for ent, idx in ent2id.items():
            f.write(f"{ent}\t{idx}\n")

    # 写 relation2id.txt
    with open(os.path.join(TRANSE_DIR, 'relation2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(relation_list)}\n")
        for rel, idx in rel2id.items():
            f.write(f"{rel}\t{idx}\n")

    # 写 train2id.txt
    with open(os.path.join(TRANSE_DIR, 'train2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(raw_data)}\n")
        for item in raw_data:
            h = ent2id[item['h_id']]
            t = ent2id[item['t_id']]
            r = rel2id[item['r_val']]
            f.write(f"{h}\t{t}\t{r}\n")

    print(f"TransE 数据已保存至: {TRANSE_DIR}")
    print(f" - 实体数: {len(entity_list)}")
    print(f" - 关系数: {len(relation_list)}")

    # ==========================
    # 5. 生成 LLM 微调数据
    # ==========================
    print("\n[2/2] 正在生成 LLM 微调数据 (Instruction Tuning)...")

    sft_dataset = []
    for item in raw_data:
        # 构造 Prompt
        # Instruction: 任务说明
        # Input: 头实体详细信息 + 关系类型 + (可选: TransE 召回的候选集，这里先留空，推理时再加)
        # Output: 尾实体详细信息

        prompt_input = (
            f"基于实体属性和图谱逻辑，预测目标实体。\n"
            f"输入实体：{item['h_desc']}\n"
            f"关系类型：{item['r_val']}\n"
            f"请分析输入实体的属性（如时间、位置、类型），推断最可能的输出实体。"
        )

        sft_dataset.append({
            "instruction": "知识图谱推理任务：根据头实体及其属性补全尾实体。",
            "input": prompt_input,
            "output": item['t_desc']
        })

    out_file = os.path.join(LLM_DIR, 'kgc_train.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(sft_dataset, f, ensure_ascii=False, indent=2)

    print(f"LLM 数据已保存至: {out_file}")
    print("全部导出完成！")


if __name__ == '__main__':
    export_and_process()