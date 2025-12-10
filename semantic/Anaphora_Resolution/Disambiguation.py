# semantic/Anaphora_Resolution/Disambiguation.py
import json
import numpy as np
from pymongo import MongoClient
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity

# 【修改点 1】导入 settings
from config.settings import settings

# 【修改点 2】移除旧的 embedding_model 导入
# from config.model import embedding_model

# 【修改点 3】根据环境导入 LangChain 的 Embedding 类
# 假设使用 OllamaEmbeddings (需要安装 langchain-ollama 或 langchain-community)
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings

# 【修改点 4】初始化 embedding_model
# 使用 settings 中的配置初始化模型实例
try:
    # 修正：这里使用 settings.DISAMBIGUATION_EMBEDDING_MODEL
    embedding_model = OllamaEmbeddings(
        base_url=settings.OLLAMA_HOST,
        model=settings.DISAMBIGUATION_EMBEDDING_MODEL,
    )
except Exception as e:
    print(f"[Disambiguation] Warning: Embedding model init failed: {e}")
    embedding_model = None


def extract_all_events(collection):
    all_docs = list(collection.find({}))
    print(f"共读取 {len(all_docs)} 条文档")

    event_pool = []
    for doc in all_docs:
        sdata = doc.get("structured_data", {})
        events = sdata.get("events", [])
        for e in events:
            event_pool.append({
                "doc_id": doc["_id"],
                "event_id": e.get("event_id", ""),
                "event_name": e.get("event_name", ""),
                "time_position": e.get("time_position", {}),
                "space_position": e.get("space_position", {}),
                "relationship_characteristics": e.get("relationship_characteristics", {}),
                "attribute_characteristics": e.get("attribute_characteristics", {}),
                "emotion_characteristics": e.get("emotion_characteristics", {}),
                "evolution_characteristics": e.get("evolution_characteristics", {}),
                "raw_event": e,
            })
    print(f"共提取 {len(event_pool)} 个子事件")
    return event_pool


def name_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def merge_event_info(ev_keep, ev_merge):
    """合并两个事件的信息，非空覆盖"""
    for key in ["time_position", "space_position", "attribute_characteristics"]:
        if not ev_keep.get(key) and ev_merge.get(key):
            ev_keep[key] = ev_merge[key]
    return ev_keep


def build_event_text(ev):
    """构造用于向量化的事件描述，包含 event_name + 关键关系字段"""
    parts = [ev.get("event_name", "")]
    rel_chars = ev.get("relationship_characteristics", {})
    if isinstance(rel_chars, dict):
        for role, rel in rel_chars.items():
            if isinstance(rel, dict):
                name = rel.get("name", "")
                relation = rel.get("relation", "")
                if name or relation:
                    parts.append(f"{role}:{name}({relation})")
    return " ".join(parts).strip()


def final_id_mapping(id_mapping):
    """展开链式映射"""
    final_map = {}
    for k in id_mapping:
        v = id_mapping[k]
        while v in id_mapping:
            v = id_mapping[v]
        final_map[k] = v
    return final_map


def remove_exact_duplicates(event_pool):
    """删除事件名称完全相同的子事件（保留第一条）"""
    unique = {}
    duplicates = []
    for ev in event_pool:
        # 使用关键字段做指纹
        key = json.dumps({
            "event_name": ev["event_name"],
            "time_position": ev["time_position"],
            "space_position": ev["space_position"],
            "attribute_characteristics": ev["attribute_characteristics"]
        }, sort_keys=True, ensure_ascii=False)

        if key in unique:
            duplicates.append(ev)
        else:
            unique[key] = ev

    print(f"删除重复事件 {len(duplicates)} 条")
    return list(unique.values()), duplicates


def main(collection_name="interim"):
    """运行指代消解"""
    # 【修改点 5】使用 settings.MONGO_URI 连接
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]
    collection = db[collection_name]

    # 提取所有子事件
    event_pool = extract_all_events(collection)
    if not event_pool:
        print("没有事件，退出")
        return

    # 删除完全重复的事件
    event_pool, duplicates = remove_exact_duplicates(event_pool)

    # 生成相似度文本（事件名称 + 关系字段）
    print("计算相似度矩阵...")
    texts = [build_event_text(ev) for ev in event_pool]

    if embedding_model is None:
        print("❌ 嵌入模型未初始化，无法计算相似度。")
        return

    try:
        embeddings = embedding_model.embed_documents(texts)
    except Exception as e:
        print(f"❌ 向量化失败: {e}")
        return

    for i, ev in enumerate(event_pool):
        ev["embedding"] = embeddings[i]

    vectors = np.array(embeddings)
    if len(vectors) == 0:
        print("无向量数据")
        return

    sim_matrix = cosine_similarity(vectors)

    # 构建ID映射
    id_mapping = {}
    merge_details = {}  # 记录修改字段
    n = len(event_pool)
    for i in range(n):
        for j in range(i + 1, n):
            # 如果 j 已经被合并过，跳过 (简单的贪婪策略)
            if id_mapping.get(event_pool[j]["event_id"]):
                continue

            name_sim = name_similarity(event_pool[i]["event_name"], event_pool[j]["event_name"])
            if name_sim < 0.7:  # 名称差太大直接跳过
                continue

            if sim_matrix[i][j] >= 0.9:  # 向量相似度阈值
                # 检查时间和地点一致性
                t1, t2 = event_pool[i]["time_position"], event_pool[j]["time_position"]
                s1, s2 = event_pool[i]["space_position"], event_pool[j]["space_position"]

                # 简单的冲突检测：如果都有值且不相等则不合并
                if (t1 and t2 and t1 != t2) or (s1 and s2 and s1 != s2):
                    continue

                # 选信息最多的事件作为保留
                info_count_i = sum(bool(v) for v in [
                    t1, s1, event_pool[i]["attribute_characteristics"],
                    event_pool[i].get("relationship_characteristics")])
                info_count_j = sum(bool(v) for v in [
                    t2, s2, event_pool[j]["attribute_characteristics"],
                    event_pool[j].get("relationship_characteristics")])

                if info_count_i >= info_count_j:
                    keep_idx, merge_idx = i, j
                else:
                    keep_idx, merge_idx = j, i

                ev_keep = event_pool[keep_idx]
                ev_merge = event_pool[merge_idx]

                merge_details[ev_merge["event_id"]] = {
                    "merged_into": ev_keep["event_id"],
                    "before_name": ev_merge["event_name"]
                }

                # 合并信息到保留对象
                event_pool[keep_idx] = merge_event_info(ev_keep, ev_merge)
                # 记录映射关系
                id_mapping[ev_merge["event_id"]] = ev_keep["event_id"]

    # 展开链式映射
    id_mapping = final_id_mapping(id_mapping)

    # 应用合并后的ID
    for ev in event_pool:
        ev["event_id"] = id_mapping.get(ev["event_id"], ev["event_id"])

    # 更新数据库
    print("应用ID合并更新数据库...")
    doc_updates = {}
    for ev in event_pool:
        old_id = ev["raw_event"]["event_id"]
        # 找到最终 ID
        curr_id = old_id
        while curr_id in id_mapping:
            curr_id = id_mapping[curr_id]

        if old_id != curr_id:
            doc_id = ev["doc_id"]
            if doc_id not in doc_updates:
                doc_updates[doc_id] = {}
            doc_updates[doc_id][old_id] = curr_id

    # 批量更新文档
    count = 0
    for doc_id, replace_map in doc_updates.items():
        doc = collection.find_one({"_id": doc_id})
        if not doc:
            continue
        sdata = doc.get("structured_data", {})
        events = sdata.get("events", [])
        relations = sdata.get("event_relations", [])

        # 替换事件ID
        new_events = []
        for e in events:
            old_eid = e.get("event_id")
            # 如果该事件ID在映射表中，替换它
            if old_eid in replace_map:
                e["event_id"] = replace_map[old_eid]

            # 过滤无效事件 (attribute_characteristics 为空)
            ac = e.get("attribute_characteristics", {})
            if not ac or ac in [None, {}, []]:
                continue
            new_events.append(e)

        # 替换关系中的ID并去重
        pair_to_relation = {}
        for r in relations:
            parts = r.strip().split()
            if len(parts) != 3:
                continue
            src, tgt, rtype = parts
            src = replace_map.get(src, src)
            tgt = replace_map.get(tgt, tgt)
            if src == tgt:
                continue
            pair_key = (src, tgt)
            if pair_key not in pair_to_relation:
                pair_to_relation[pair_key] = rtype

        sdata["events"] = new_events
        sdata["event_relations"] = [f"{src}  {tgt}  {rtype}" for (src, tgt), rtype in pair_to_relation.items()]

        collection.update_one({"_id": doc_id}, {"$set": {"structured_data": sdata}})
        count += 1

    # 输出日志
    try:
        with open("detail_oref_merge_log.json", "w", encoding="utf-8") as f:
            json.dump({
                "id_mapping": id_mapping,
                "merge_details": merge_details
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(f"✅ 指代消解完成，更新了 {count} 条文档。")


if __name__ == "__main__":
    main()