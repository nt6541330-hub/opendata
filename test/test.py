# 文件路径: open_source_data/kg_completion/fetch_test_json.py
import json
import random
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

NEBULA_IP = "39.104.200.88"
NEBULA_PORT = 41003
USER = "root"
PASSWORD = "123456"
SPACE_NAME = "event_target1"


def fetch_and_generate_json():
    print(">>> 正在连接 NebulaGraph...")
    config = Config()
    config.max_connection_pool_size = 5
    pool = ConnectionPool()
    if not pool.init([(NEBULA_IP, NEBULA_PORT)], config):
        print("无法连接到 NebulaGraph")
        return

    session = pool.get_session(USER, PASSWORD)
    session.execute(f'USE {SPACE_NAME}')

    print(">>> 正在随机采样数据 (获取 properties(e).relation)...")
    # === 核心修正：查询 relation 属性 ===
    gql = """
    MATCH (h)-[e]->(t)
    RETURN 
        id(h) as h_id, properties(h).name as h_name, properties(h).event_name as h_evt_name,
        properties(e).relation as rel_prop,
        type(e) as rel_type,
        id(t) as t_id, properties(t).name as t_name, properties(t).event_name as t_evt_name
    LIMIT 15;
    """

    result = session.execute(gql)
    if not result.is_succeeded():
        print(f"查询失败: {result.error_msg()}")
        return

    graph_data = []
    size = result.row_size()
    for i in range(size):
        row = result.row_values(i)

        h_val = row[0].as_string()

        # 优先取 relation 属性
        r_prop = row[3].as_string() if not row[3].is_null() else None
        r_type = row[4].as_string()
        final_r = r_prop if r_prop else r_type

        t_val = row[5].as_string()

        # 提取名称
        h_name = row[1].as_string() if not row[1].is_null() else ""
        h_evt = row[2].as_string() if not row[2].is_null() else ""
        h_label = h_name or h_evt or h_val

        t_name = row[6].as_string() if not row[6].is_null() else ""
        t_evt = row[7].as_string() if not row[7].is_null() else ""
        t_label = t_name or t_evt or t_val

        graph_data.append({
            "h": h_label,
            "r": final_r,  # 这里现在是细粒度关系了
            "t": t_label
        })

    payload = {
        "graph": graph_data,
        "params": {
            "which": "all",
            "query_space": "observed",
            "conf": 0.2,
            "topk_recall": 120,
            "topm_rerank": 20,
            "top_degree_k": 0,
            "sample_rate": 1.0,
            "budget_seconds": 0,
            "max_return": 5,
            "llm_only_on_oov": False
        }
    }

    print("\n" + "=" * 20 + " 生成的测试 JSON " + "=" * 20)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("=" * 60)

    session.release()
    pool.close()


if __name__ == "__main__":
    fetch_and_generate_json()