import json
import random
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

# === 配置信息 ===
NEBULA_IP = "39.104.200.88"
NEBULA_PORT = 41003
USER = "root"
PASSWORD = "123456"
SPACE_NAME = "event_target1"


def fetch_and_generate_json():
    # 1. 连接数据库
    config = Config()
    config.max_connection_pool_size = 5
    pool = ConnectionPool()
    if not pool.init([(NEBULA_IP, NEBULA_PORT)], config):
        print("无法连接到 NebulaGraph")
        return

    session = pool.get_session(USER, PASSWORD)
    session.execute(f'USE {SPACE_NAME}')

    # 2. 随机采样查询
    # 获取 15 条有关系的边，并尝试获取 name 或 event_name 属性
    print("正在从数据库采样数据...")

    # 这里的查询尝试获取边的类型，以及起止点的名称
    # 如果点是事件，通常有 event_name；如果是人/组织，通常有 name
    gql = """
    MATCH (h)-[e]->(t)
    RETURN 
        id(h) as h_id, properties(h).name as h_name, properties(h).event_name as h_evt_name,
        type(e) as r_type,
        id(t) as t_id, properties(t).name as t_name, properties(t).event_name as t_evt_name
    LIMIT 15;
    """

    result = session.execute(gql)
    if not result.is_succeeded():
        print(f"查询失败: {result.error_msg()}")
        return

    graph_data = []

    # 3. 处理结果
    size = result.row_size()
    for i in range(size):
        row = result.row_values(i)

        # 获取 ID
        h_val = row[0].as_string()
        t_val = row[4].as_string()
        r_val = row[3].as_string()  # 关系名

        # 尝试获取更友好的名称 (优先用 name/event_name，没有就用 ID)
        # Nebula 返回的是 Value 对象，需要判断是否为空
        h_name_prop = row[1].as_string() if not row[1].is_null() else ""
        h_evt_prop = row[2].as_string() if not row[2].is_null() else ""

        t_name_prop = row[5].as_string() if not row[5].is_null() else ""
        t_evt_prop = row[6].as_string() if not row[6].is_null() else ""

        # 决定 h 的显示名称
        final_h = h_name_prop if h_name_prop else (h_evt_prop if h_evt_prop else h_val)
        # 决定 t 的显示名称
        final_t = t_name_prop if t_name_prop else (t_evt_prop if t_evt_prop else t_val)

        # 构造三元组
        graph_data.append({
            "h": final_h,
            "r": r_val,  # 这里保持数据库原本的英文关系名，或者你可以手动映射
            "t": final_t
        })

    # 4. 构造最终请求体
    payload = {
        "graph": graph_data,
        "params": {
            "which": "all",
            "query_space": "observed",
            "conf": 0.2,  # 融合分数阈值
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