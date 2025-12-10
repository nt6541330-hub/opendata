# semantic/Abstract/abstract.py
# -*- coding: utf-8 -*-

from pymongo import MongoClient
from bson import ObjectId, errors as bson_errors
import requests
import re
from typing import Optional

from config.settings import settings


# =========================
# LLM 摘要函数
# =========================
def generate_abstract_llm(text: str, max_len: int = None) -> str:
    """调用本地 Ollama 生成不超过两句话的摘要"""
    if max_len is None:
        max_len = settings.MAX_ABSTRACT_LEN

    if not text:
        return ""

    if len(text) > settings.MAX_PROMPT_INPUT_LEN:
        text = text[:settings.MAX_PROMPT_INPUT_LEN]

    prompt = (
            "请阅读以下长文本内容，并将其浓缩为不超过两句话的摘要，要求：\n"
            "1. 准确传达核心信息；\n"
            "2. 表达简洁、客观，无评论；\n"
            "3. 不超过200字，只输出结果。\n\n"
            "【正文开始】\n" + text + "\n【正文结束】\n\n"
                                "请直接输出摘要："
    )

    try:
        resp = requests.post(
            settings.ABSTRACT_OLLAMA_URL,
            json={"model": settings.ABSTRACT_OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=settings.HTTP_TIMEOUT
        )
    except Exception as e:
        print(f"[Abstract] LLM请求异常: {e}", flush=True)
        return ""

    if resp.status_code != 200:
        print(f"[Abstract] LLM HTTP Error {resp.status_code}: {resp.text[:100]}", flush=True)
        return ""

    try:
        result = resp.json()
    except Exception:
        return ""

    raw_output = (result.get("response") or "").strip()

    # 清洗
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_output, flags=re.DOTALL)
    cleaned = re.sub(
        r"^(摘要[:：]|我是.*?助手[:：]?|以下是.*?摘要[:：]?|总结如下[:：]?|请看摘要[:：]?|一两句话总结如下[:：]?)",
        "",
        cleaned,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\n+", " ", cleaned).strip()

    return cleaned[:max_len] if cleaned else ""


def find_source_doc(src_id, source_collections) -> Optional[dict]:
    """在源集合中查找文档"""
    for name, col in source_collections.items():
        try:
            # 1. 尝试 ObjectId
            if isinstance(src_id, ObjectId):
                doc = col.find_one({"_id": src_id}, {"content": 1, "title": 1})
                if doc: return doc

            # 2. 尝试转 ObjectId
            try:
                oid = ObjectId(str(src_id))
                doc = col.find_one({"_id": oid}, {"content": 1, "title": 1})
                if doc: return doc
            except bson_errors.InvalidId:
                pass

            # 3. 尝试字符串 ID
            doc = col.find_one({"_id": str(src_id)}, {"content": 1, "title": 1})
            if doc: return doc

        except Exception:
            continue
    return None


def main():
    print("[Abstract] 开始检查并生成摘要...", flush=True)
    client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[settings.MONGO_DB_NAME]

    event_col = db[settings.EVENT_NODE_COLLECTION]

    # 【优化】除了源集合，也加入 interim 集合作为查找源，防止测试数据找不到
    source_collections = {name: db[name] for name in settings.COL_SRC_LIST}
    if settings.COLL_INTERIM:
        source_collections[settings.COLL_INTERIM] = db[settings.COLL_INTERIM]

    # 查询缺少摘要的记录
    need_query = {
        "$and": [
            {"$or": [
                {"abstract": {"$exists": False}},
                {"abstract": None},
                {"abstract": ""},
                {"abstract": {"$regex": r"^\s*$"}}
            ]},
            {"source_doc_id": {"$exists": True}}
        ]
    }

    # 使用 session 防止游标超时警告
    with client.start_session() as session:
        cursor = event_col.find(need_query, {"source_doc_id": 1, "event_name": 1}, no_cursor_timeout=True,
                                session=session)
        processed = 0
        skipped = 0

        try:
            for doc in cursor:
                eid = doc["_id"]
                src_id = doc.get("source_doc_id")
                ev_name = doc.get("event_name", "Unknown")

                source_doc = find_source_doc(src_id, source_collections)
                if not source_doc:
                    # print(f"[跳过] 源文档未找到: event={ev_name}, src_id={src_id}")
                    skipped += 1
                    continue

                # 【优化】同时尝试获取 content 和 title
                content = (source_doc.get("content") or source_doc.get("title") or "").strip()
                if not content:
                    # print(f"[跳过] 源文档内容为空: event={ev_name}")
                    skipped += 1
                    continue

                abstract_text = generate_abstract_llm(content)
                if abstract_text:
                    event_col.update_one({"_id": eid}, {"$set": {"abstract": abstract_text}})
                    processed += 1
                    # print(f"[成功] 生成摘要: {ev_name[:10]}...")
                else:
                    # print(f"[跳过] LLM 生成为空: event={ev_name}")
                    skipped += 1

        except Exception as e:
            print(f"[Abstract] 循环异常: {e}")
        finally:
            cursor.close()

    print(f"[Abstract] 完成，共更新 {processed} 条摘要，跳过 {skipped} 条。")


if __name__ == "__main__":
    main()