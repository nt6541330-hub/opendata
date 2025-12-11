# semantic/Into_mongodb/mogongdb.py
import os
import csv
import json
import re
from datetime import datetime
from collections import defaultdict

from pymongo import MongoClient
from bson import ObjectId

# 【修改点 1】引入 settings，移除 config.relation_whitelist
from config.settings import settings

# --- 文件输出目录（CSV / JSON）---
# 使用 settings.BASE_DIR 相对路径，避免硬编码
CSV_DIR = os.path.join(settings.BASE_DIR, "csv_output")
os.makedirs(CSV_DIR, exist_ok=True)

EVENT_EVENT_EDGE_FILE = os.path.join(CSV_DIR, "event_event_edges.csv")  # 事件-事件边
EVENT_GOAL_EDGE_FILE = os.path.join(CSV_DIR, "event_goal.csv")  # 事件-目标边（中间文件）

SUCCESS_CSV = os.path.join(CSV_DIR, "success.csv")  # 映射成功的事件-目标边（dst 已是 target_id）
FAILED_CSV = os.path.join(CSV_DIR, "failed.csv")  # 映射失败的事件-目标边
SUCCESS_JSON = os.path.join(CSV_DIR, "success.json")  # 映射成功涉及的目标节点信息（去重）

TARGET_TARGET_EDGE_FILE = os.path.join(CSV_DIR, "target_target_edges.csv")  # 目标-目标边
TARGET_TARGET_FAILED_FILE = os.path.join(CSV_DIR, "target_target_edges_failed.csv")  # 目标-目标边匹配失败记录

# --- 根据事件-目标关系推断目标类型 → 对应 knowledge_target 的 event_second_level ---
TYPE_TO_EVENT_LEVEL = {
    "character": "人物目标",
    "organization": "组织目标",
    "facility": ["机场目标", "港口目标"],
    "unknow": ["人物目标", "组织目标"],  # 兜底
}


# ============================================================
# 工具函数
# ============================================================

def is_event_id(x: str) -> bool:
    """判断一个 ID 是否是事件 ID：Th_ / Th- / TH_ / TH- 开头"""
    x = (x or "").strip()
    upper = x.upper()
    return upper.startswith("TH_") or upper.startswith("TH-")


def infer_target_type_from_relation(rel: str) -> str:
    """根据 relation 推断目标类型：organization / character / facility / unknow"""
    rel = (rel or "").lower()
    if "organization" in rel:
        return "organization"
    if "person" in rel:
        return "character"
    if "facility" in rel:
        return "facility"
    return "unknow"


def json_converter(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, ObjectId):
        return str(obj)
    return str(obj)


def load_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def collect_fieldnames(rows):
    fields = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    return fields


def save_csv(file_path, rows):
    if not rows:
        return
    fieldnames = collect_fieldnames(rows)
    with open(file_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(file_path, data):
    if not data:
        return
    with open(file_path, mode='w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_converter)


def split_alias(alias_str):
    """拆分别名：兼容 顿号/逗号/分号/竖线 等"""
    if not alias_str:
        return []
    seps = ["、", ",", "，", ";", "；", "|", " ", " "]
    tmp = [alias_str]
    for s in seps:
        alias_list = []
        for part in tmp:
            alias_list.extend(part.split(s))
        tmp = alias_list
    return [x.strip() for x in tmp if x and x.strip()]


def flatten_dict(d, parent_key='', sep='.'):
    """展平嵌套字典 / 列表"""
    items = []
    if not isinstance(d, dict):
        return {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for idx, sub_v in enumerate(v):
                if isinstance(sub_v, dict):
                    items.extend(flatten_dict(sub_v, f"{new_key}{sep}{idx}", sep=sep).items())
                else:
                    items.append((f"{new_key}{sep}{idx}", sub_v))
        else:
            items.append((new_key, v))
    return dict(items)


def build_name_alias_index(event_second_level, collection_knowledge):
    """
    在 knowledge_target 上构建 name/alias → (target_id, data) 索引
    """
    index = {}

    if isinstance(event_second_level, list):
        query = {"event_second_level": {"$in": event_second_level}}
    else:
        query = {"event_second_level": event_second_level}

    for doc in collection_knowledge.find(query):
        data = doc.get("data", {})
        target_id = doc.get("target_id", "")

        if isinstance(data, dict):
            name = (data.get("name") or "").strip()
            alias = (data.get("alias") or "").strip()
            if name and name not in index:
                index[name] = (target_id, data)
            if alias:
                for a in split_alias(alias):
                    if a and a not in index:
                        index[a] = (target_id, data)

        elif isinstance(data, list):
            for record in data:
                if isinstance(record, dict):
                    name = (record.get("name") or "").strip()
                    alias = (record.get("alias") or "").strip()
                    tid = record.get("target_id", target_id)
                    if name and name not in index:
                        index[name] = (tid, record)
                    if alias:
                        for a in split_alias(alias):
                            if a and a not in index:
                                index[a] = (tid, record)

    return index


def match_value_to_ids(value, index):
    """把一个字段值映射到若干 target_id"""
    hits = []
    if not value:
        return hits

    parts = split_alias(value)
    if not parts:
        parts = [str(value).strip()]

    for token in parts:
        token = token.strip()
        if not token:
            continue
        match = index.get(token)
        if match:
            tid, _ = match
            if tid not in hits:
                hits.append(tid)
    return hits


def normalize_leader_name(raw):
    """去除职务说明，例如：'孙力（国际调解院筹备办公室主任）' -> '孙力'"""
    if not raw:
        return ""
    name = re.split(r"[（(]", str(raw), 1)[0]
    return name.strip()


# ============================================================
# 步骤 1：从 interim 提取 & 清洗事件边
# ============================================================

def extract_and_clean_edges(collection_interim):
    """
    从 MongoDB 读取 structured_data.event_relations，拆分并清洗边
    """
    print("🔍 正在从 MongoDB 提取 structured_data.event_relations ...")

    event_event_edges = []
    event_goal_edges = []
    removed_count = 0

    seen_event_event = set()
    seen_event_goal = set()

    cursor = collection_interim.find(
        {},
        {"_id": 0, "structured_data.event_relations": 1}
    )

    for doc in cursor:
        relations = doc.get("structured_data", {}).get("event_relations", [])

        for relation_str in relations:
            if not relation_str:
                continue

            parts = relation_str.strip().split()
            if len(parts) != 3:
                removed_count += 1
                continue

            src, dst, rel = parts
            src = src.strip()
            dst = dst.strip()
            rel = rel.strip()

            if not src or not dst or not rel:
                removed_count += 1
                continue

            if src == dst:
                removed_count += 1
                continue

            src_is_event = is_event_id(src)
            dst_is_event = is_event_id(dst)

            # ---- 事件-事件边 ----
            # 【修改点 2】使用 settings.ALLOWED_EVENT_EVENT_RELATIONS
            if src_is_event and dst_is_event and rel in settings.ALLOWED_EVENT_EVENT_RELATIONS:
                key = (src, rel, dst)
                if key not in seen_event_event:
                    seen_event_event.add(key)
                    event_event_edges.append({
                        "src_event_id": src,
                        "relation": rel,
                        "dst_event_id": dst,
                    })
                continue

            # ---- 事件-目标边 ----
            # 【修改点 3】使用 settings.ALLOWED_EVENT_TARGET_RELATIONS
            if src_is_event and (not dst_is_event) and rel in settings.ALLOWED_EVENT_TARGET_RELATIONS:
                key = (src, rel, dst)
                if key not in seen_event_goal:
                    seen_event_goal.add(key)
                    target_type = infer_target_type_from_relation(rel)
                    event_goal_edges.append({
                        "src_id": src,
                        "relation": rel,
                        "dst_id": dst,
                        "type": target_type,
                    })
                continue

            # 其他情况丢弃
            removed_count += 1

    with open(EVENT_EVENT_EDGE_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["src_event_id", "relation", "dst_event_id"])
        writer.writeheader()
        writer.writerows(event_event_edges)

    with open(EVENT_GOAL_EDGE_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["src_id", "relation", "dst_id", "type"])
        writer.writeheader()
        writer.writerows(event_goal_edges)

    print("✅ 提取 & 初步清洗完成：")
    print(f"   · 事件-事件边：{len(event_event_edges)} 条")
    print(f"   · 事件-目标边：{len(event_goal_edges)} 条")
    print(f"   · 丢弃无效记录：{removed_count} 条")


# ============================================================
# 步骤 2：对事件-目标边就地去重
# ============================================================

def dedup_event_goal_edges_inplace():
    """对 event_goal.csv 按 (src_id, dst_id) 去重"""
    if not os.path.exists(EVENT_GOAL_EDGE_FILE):
        print(f"⚠️ 未找到事件目标边文件：{EVENT_GOAL_EDGE_FILE}，跳过去重。")
        return

    print("🧹 正在对事件-目标边按 (src_id, dst_id) 去重...")

    rows = load_csv(EVENT_GOAL_EDGE_FILE)
    seen_pairs = set()
    cleaned = []
    duplicates = []

    for r in rows:
        src = (r.get("src_id") or "").strip()
        dst = (r.get("dst_id") or "").strip()
        key = (src, dst)
        if key in seen_pairs:
            duplicates.append(r)
        else:
            seen_pairs.add(key)
            cleaned.append(r)

    if duplicates:
        print("🔁 检测到重复的事件-目标边，将被去重。")
        save_csv(EVENT_GOAL_EDGE_FILE, cleaned)
        print(f"✅ 去重完成，保存到：{EVENT_GOAL_EDGE_FILE}")
    else:
        print("✅ 无需去重。")


# ============================================================
# 步骤 3：事件-目标边映射到 knowledge_target
# ============================================================

def map_event_goal_edges(collection_knowledge):
    """
    映射 dst_id 到 target_id，生成 success.csv 和 failed.csv
    """
    if not os.path.exists(EVENT_GOAL_EDGE_FILE):
        print(f"⚠️ 未找到事件目标边文件，无法映射。")
        return

    rows = load_csv(EVENT_GOAL_EDGE_FILE)

    # 预构建索引
    indices = {}
    for t, ev in TYPE_TO_EVENT_LEVEL.items():
        print(f"构建索引：type={t}, event_second_level={ev}")
        indices[t] = build_name_alias_index(ev, collection_knowledge)

    success_rows_all = []
    failed_rows_all = []
    success_json_all = []

    json_seen_target_ids = set()
    existing_pairs_by_type = defaultdict(set)

    for line_no, row in enumerate(rows, start=1):
        dst_name = (row.get("dst_id") or "").strip()
        type_raw = (row.get("type") or "").strip()
        src_id_val = (row.get("src_id") or "").strip()
        relation_val = row.get("relation")

        if type_raw not in TYPE_TO_EVENT_LEVEL:
            failed_rows_all.append(row)
            continue

        index = indices.get(type_raw) or {}
        alias_parts = split_alias(dst_name)
        if not alias_parts:
            alias_parts = [dst_name]

        match_found = False

        for part in alias_parts:
            if not part: continue
            match = index.get(part)
            if not match: continue

            target_id, data = match
            pair = (src_id_val, target_id)

            if pair in existing_pairs_by_type[type_raw]:
                continue

            new_row = dict(row)
            new_row["dst_id"] = target_id
            if "type" in new_row:
                del new_row["type"]
            success_rows_all.append(new_row)

            if target_id not in json_seen_target_ids:
                success_json_all.append({
                    "target_id": target_id,
                    "name": dst_name,
                    "data": data,
                })
                json_seen_target_ids.add(target_id)

            existing_pairs_by_type[type_raw].add(pair)
            match_found = True
            break

        if not match_found:
            failed_rows_all.append(row)

    if success_rows_all:
        save_csv(SUCCESS_CSV, success_rows_all)
        print(f"✅ 成功记录保存到：{SUCCESS_CSV} (共 {len(success_rows_all)} 条)")

    if failed_rows_all:
        save_csv(FAILED_CSV, failed_rows_all)
        print(f"⚠️ 失败记录保存到：{FAILED_CSV} (共 {len(failed_rows_all)} 条)")

    if success_json_all:
        save_json(SUCCESS_JSON, success_json_all)
        print(f"📄 成功 JSON 保存到：{SUCCESS_JSON}")


# ============================================================
# 步骤 4：目标-目标边抽取
# ============================================================

def extract_target_target_edges(collection_knowledge):
    """
    抽取目标之间的关系边
    """
    if os.path.exists(SUCCESS_JSON):
        with open(SUCCESS_JSON, "r", encoding="utf-8") as f:
            success_records = json.load(f)
        allowed_src_ids = {str(rec.get("target_id")) for rec in success_records if rec.get("target_id")}
        print(f"📦 从 success.json 读取到 {len(allowed_src_ids)} 个目标作为 src。")
    else:
        allowed_src_ids = None
        print(f"⚠️ 未找到 success.json，全量抽取。")

    print("🔍 构建各类目标索引...")
    person_index = build_name_alias_index("人物目标", collection_knowledge)
    org_index = build_name_alias_index("组织目标", collection_knowledge)

    edges_seen = set()

    with open(TARGET_TARGET_EDGE_FILE, "w", newline="", encoding="utf-8-sig") as edges_f, \
            open(TARGET_TARGET_FAILED_FILE, "w", newline="", encoding="utf-8-sig") as failed_f:

        edges_writer = csv.DictWriter(edges_f, fieldnames=["src_id", "relation", "dst_id"])
        failed_writer = csv.DictWriter(failed_f, fieldnames=["src_id", "relation", "dst_raw"])
        edges_writer.writeheader()
        failed_writer.writeheader()

        total_edges = 0

        query = {"target_id": {"$in": list(allowed_src_ids)}} if allowed_src_ids else {}

        print("🔄 正在抽取目标-目标边...")

        for doc in collection_knowledge.find(query):
            src_id = doc.get("target_id")
            ev2 = doc.get("event_second_level")
            data = doc.get("data", {}) or {}

            if not src_id or not ev2:
                continue

            # 辅助函数：处理单个关系字段
            def process_relation(field_name, relation_name, target_index):
                nonlocal total_edges
                raw_val = data.get(field_name)
                if raw_val:
                    # 特殊处理：leader 字段去职务
                    search_val = normalize_leader_name(raw_val) if field_name == "leader" else raw_val
                    dst_ids = match_value_to_ids(search_val, target_index)
                    if dst_ids:
                        for dst in dst_ids:
                            key = (src_id, relation_name, dst)
                            if key not in edges_seen:
                                edges_seen.add(key)
                                edges_writer.writerow({"src_id": src_id, "relation": relation_name, "dst_id": dst})
                                total_edges += 1
                    else:
                        failed_writer.writerow({"src_id": src_id, "relation": relation_name, "dst_raw": raw_val})

            if ev2 == "人物目标":
                process_relation("affiliated_organization", "person-personOrganization-affiliated", org_index)
            elif ev2 == "组织目标":
                process_relation("superior_organization", "organization-organization-superior", org_index)
                process_relation("subordinate_units", "organization-organization-subordinate", org_index)
                process_relation("international_cooperation", "organization-organization-joint", org_index)
                process_relation("leader", "target-target-position-leader", person_index)
            elif ev2 == "机场目标":
                process_relation("operation_operator", "target-target-operation-operatedBy", org_index)
            elif ev2 == "港口目标":
                process_relation("operator", "target-target-operation-operatedBy", org_index)
            elif ev2 == "军舰目标":
                process_relation("command_unit", "target-target-operation-commandedBy", org_index)

        print(f"✅ 目标-目标边抽取完成：写入 {total_edges} 条边")


# ============================================================
# 步骤 5：CSV 入库到 EDGES_COLLECTION
# ============================================================

def import_edges_to_mongo(db):
    """
    将所有生成的边导入 MongoDB
    """
    collection = db[settings.EDGES_COLLECTION]
    collection.delete_many({})
    print(f"🧹 已清空 {settings.EDGES_COLLECTION} 集合")

    def load_and_insert(file_path, name):
        count = 0
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 根据不同文件映射字段名
                    src = row.get("src_id") or row.get("src_event_id")
                    dst = row.get("dst_id") or row.get("dst_event_id")
                    rel = row.get("relation")
                    if src and dst and rel:
                        collection.insert_one({
                            "src": src.strip(),
                            "dst": dst.strip(),
                            "relation": rel.strip()
                        })
                        count += 1
        print(f"✅ 导入 {count} 条 {name}")

    load_and_insert(SUCCESS_CSV, "事件-目标 边")
    load_and_insert(EVENT_EVENT_EDGE_FILE, "事件-事件 边")
    load_and_insert(TARGET_TARGET_EDGE_FILE, "目标-目标 边")


# ============================================================
# 步骤 6：从边集合提取事件节点并展平入库
# ============================================================

def extract_event_nodes_from_edges(db):
    """
    提取涉及的事件节点并入库
    """
    source_col = db[settings.INTERIM_COLLECTION]
    edges_col = db[settings.EDGES_COLLECTION]
    final_nodes_col = db[settings.EVENT_NODE_COLLECTION]

    event_ids_from_edges = set()
    for edge in edges_col.find({}, {"src": 1, "dst": 1}):
        src = edge.get("src")
        dst = edge.get("dst")
        if src and src.startswith("Th"): event_ids_from_edges.add(src)
        if dst and dst.startswith("Th"): event_ids_from_edges.add(dst)

    print(f"🔎 找到 {len(event_ids_from_edges)} 个唯一的事件ID")

    # 不再清空集合，只统计新增 / 更新
    insert_count = 0
    update_count = 0

    for event_id_to_find in event_ids_from_edges:
        source_doc = source_col.find_one({
            "structured_data.events.event_id": event_id_to_find
        })

        if not source_doc:
            continue

        doc_id = str(source_doc.get("_id"))
        source = source_doc.get("source", "")
        first_level = source_doc.get("event_first_level", "")
        second_level = source_doc.get("event_second_level", "")

        structured_events = source_doc.get("structured_data", {}).get("events", [])
        event_data = next((e for e in structured_events if e.get("event_id") == event_id_to_find), None)

        if not event_data:
            continue

        flat = {
            "source_doc_id": doc_id,
            "source": source,
            "event_first_level": first_level,
            "event_second_level": second_level,
            "event_id": event_id_to_find,
            "event_name": event_data.get("event_name", ""),
            "status": "0"
        }

        # 展平各个子字段
        time_pos = event_data.get("time_position", {})
        if isinstance(time_pos, dict):
            for k, v in time_pos.items(): flat[f"time_position_{k}"] = v

        space_pos = event_data.get("space_position", {})
        if isinstance(space_pos, dict):
            for k, v in space_pos.items(): flat[f"space_position_{k}"] = v

        rel = event_data.get("relationship_characteristics", {})
        if isinstance(rel, dict):
            for role, role_info in rel.items():
                if isinstance(role_info, dict):
                    name = role_info.get("name")
                    if name: flat[f"relationship_characteristics_{role}"] = name

        attr = event_data.get("attribute_characteristics", {})
        if isinstance(attr, dict):
            for k, v in attr.items(): flat[f"attribute_{k}"] = v

        emo = event_data.get("emotion_characteristics", {})
        if isinstance(emo, dict) and emo:
            flat["emotion_characteristics"] = "；".join(emo.keys())

        # 以 event_id 为唯一键做 upsert
        result = final_nodes_col.update_one(
            {"event_id": event_id_to_find},  # 查询条件
            {"$set": flat},  # 更新内容
            upsert=True  # 不存在就插入
        )

        # 统计是新增还是更新
        if result.matched_count == 0:
            insert_count += 1  # 新插入
        else:
            update_count += 1  # 已存在，执行了更新

    print(f"✅ 成功将 {insert_count} 条事件写入 {settings.EVENT_NODE_COLLECTION}")


# ============================================================
# 步骤 7：展平目标节点 success.json → TARGET_NODE_COLLECTION
# ============================================================

def flatten_target_nodes(db, collection_knowledge):
    """
    展平涉及的目标节点信息
    """
    new_collection = db[settings.TARGET_NODE_COLLECTION]

    if not os.path.exists(SUCCESS_JSON):
        return

    with open(SUCCESS_JSON, mode='r', encoding='utf-8') as f:
        records = json.load(f)

    # 不再清空集合
    target_ids = {r.get("target_id") for r in records if r.get("target_id")}

    insert_count = 0
    update_count = 0

    type_mapping = {
        "人物目标": "人物", "组织目标": "组织", "机场目标": "机场",
        "港口目标": "港口", "军舰目标": "军舰"
    }

    for tid in target_ids:
        doc = collection_knowledge.find_one({"target_id": tid})
        if not doc: continue

        data = doc.get("data", {}) or {}
        flat_data = flatten_dict(data)

        flat_data["target_id"] = tid
        flat_data["name"] = data.get("name", doc.get("name", ""))
        flat_data["source_url"] = doc.get("source_url", "")
        flat_data["event_first_level"] = doc.get("event_first_level", "")
        flat_data["event_second_level"] = doc.get("event_second_level", "")
        flat_data["type"] = type_mapping.get(flat_data["event_second_level"], "其他")

        result = new_collection.update_one(
            {"target_id": tid},  # 以 target_id 唯一定位
            {"$set": flat_data},  # 覆盖数据
            upsert=True
        )

        if result.matched_count == 0:
            insert_count += 1
        else:
            update_count += 1

    print(f"✅ 目标节点展平完成：写入 {insert_count} 条到 {settings.TARGET_NODE_COLLECTION}")


# ============================================================
# 总入口
# ============================================================

def main():
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]

    collection_interim = db[settings.INTERIM_COLLECTION]
    collection_knowledge = db[settings.KNOWLEDGE_COLLECTION_TARGET]

    # 1. 提取 & 清洗
    extract_and_clean_edges(collection_interim)

    # 2. 去重
    dedup_event_goal_edges_inplace()

    # 3. 映射 knowledge_target
    map_event_goal_edges(collection_knowledge)

    # 4. 抽取目标关系
    extract_target_target_edges(collection_knowledge)

    # 5. 入库边
    import_edges_to_mongo(db)

    # 6. 入库事件节点
    extract_event_nodes_from_edges(db)

    # 7. 入库目标节点
    flatten_target_nodes(db, collection_knowledge)


if __name__ == "__main__":
    main()