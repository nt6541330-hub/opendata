from pymongo import MongoClient
from gridfs import GridFS
from config.settings import settings

# 全局单例
_mongo_client = None

def get_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000
        )
    return _mongo_client

def get_db():
    return get_client()[settings.MONGO_DB_NAME]

def get_collection(coll_name: str):
    return get_db()[coll_name]

def get_gridfs(bucket_name: str) -> GridFS:
    return GridFS(get_db(), collection=bucket_name)

# --- 快捷获取各平台存储对象 (Collection, GridFS) ---

def get_weibo_storage():
    return get_collection(settings.COLL_WEIBO), get_gridfs(settings.BUCKET_WEIBO)

def get_cctv_storage():
    return get_collection(settings.COLL_CCTV), get_gridfs(settings.BUCKET_CCTV)

def get_toutiao_storage():
    return get_collection(settings.COLL_TOUTIAO), get_gridfs(settings.BUCKET_TOUTIAO)

def get_xinhua_storage():
    return get_collection(settings.COLL_XINHUA), get_gridfs(settings.BUCKET_XINHUA)

def get_cnn_storage():
    return get_collection(settings.COLL_CNN), get_gridfs(settings.BUCKET_CNN)

# 【新增】获取任务日志集合
def get_task_collection():
    # 集合名暂定为 'collect_tasks'
    return get_db()['crawler_logs']