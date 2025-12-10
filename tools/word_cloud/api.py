
# open_source_data/tools/word_cloud/api.py
import os
import re
from collections import Counter
from typing import List, Optional

import jieba
from fastapi import APIRouter, Query
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from common.utils import get_logger

# 接入统一日志
logger = get_logger(__name__)

router = APIRouter()

# ----- 配置：保留原脚本的特定配置 -----
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://root:123456@39.104.200.88:41004/wiki?authSource=admin')
DB_NAME = os.getenv('DB_NAME', 'NEWS')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'source_cctv')

# ----- Mongo 连接 -----
# 模块加载时建立连接
try:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=3000,
        socketTimeoutMS=10000,
        connectTimeoutMS=3000,
    )
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
except Exception as e:
    logger.error(f"词云工具数据库连接失败: {e}")
    client = None
    collection = None

# ----- 停用词 -----
STOPWORDS = {
    '编辑', '责任编辑', '记者', '责编', '新华社', '李学仁', '韩墨',
    '杨依军', '丁林', '王佳宁', '责任', '进行', '日电', '北京', '广东',
    '我们', '一个', '指出', '对此', '发言人', '任何', '活动', '没有', '加征', '协议',
    '时间', '通过', '认为', '影响', '亿美元', '###', '相关', '代表'
}


def is_valid_word(word: str) -> bool:
    if not word or word.strip() == '':
        return False
    if word in STOPWORDS:
        return False
    if word.isdigit():
        return False
    return True


def process_text(text: str):
    # 仅保留长度>1的词
    for w in jieba.cut(text, cut_all=False):
        w = w.strip()
        if len(w) > 1 and is_valid_word(w):
            yield w


# ----- 路由定义 -----
@router.get("/generate", summary="生成热点词云")
async def generate_wordcloud(
        keyword: Optional[str] = Query(None, description="搜索关键词，逗号分隔"),
        limit: int = Query(1000, ge=1, le=5000, description="分析的数据条数限制")
):
    # 【修复点】使用 is None 进行显式判断
    if collection is None:
        return {"code": 500, "msg": "数据库未连接", "data": []}

    mongo_query = {}
    if keyword:
        # 支持多关键词
        keywords = [k.strip() for k in keyword.split(",") if k.strip()]
        keywords = keywords[:8]  # 最多 8 个关键字
        or_list = []
        for kw in keywords:
            rx = re.compile(re.escape(kw), re.IGNORECASE)
            or_list.extend([
                {"content": rx},
                {"title": rx},
                {"label": rx},
                {"keyword": rx},
            ])
        if or_list:
            mongo_query = {"$or": or_list}

    try:
        # 使用 to_list 或迭代器 (这里用迭代器以节省内存)
        cursor = collection.find(
            mongo_query,
            {'content': 1, '_id': 0}
        ).limit(limit).max_time_ms(10_000)
    except PyMongoError as e:
        logger.error(f"Mongo query error: {e}")
        return {"code": 500, "msg": f"数据库错误: {e}", "data": []}

    word_counter = Counter()
    matched_docs = 0

    # 注意：FastAPI中如果不使用 async 驱动，这里的循环会阻塞 worker
    for item in cursor:
        text = item.get('content') or ''
        if not text:
            continue
        matched_docs += 1
        word_counter.update(process_text(text))

    if matched_docs == 0 or not word_counter:
        return {"code": 200, "msg": "无匹配内容", "data": []}

    top_words = word_counter.most_common(40)
    max_count = top_words[0][1] or 1

    data_list = [
        {"name": word, "value": round(count / max_count, 4), "raw": count}
        for word, count in top_words
    ]

    return {"code": 200, "msg": "操作成功", "data": data_list}


# ----- 预热函数 -----
async def warmup():
    """预热：初始化 jieba 字典"""
    logger.info("🔥 [WordCloud] 正在加载 Jieba 字典...")
    # jieba 初始化是同步阻塞的，但在启动阶段执行一次即可
    jieba.initialize()
    logger.info("✅ [WordCloud] Jieba 字典加载完成")