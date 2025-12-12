import time
import logging
import traceback
import pymongo  # 【新增】需要导入 pymongo 以捕获特定错误
from datetime import datetime
from bson import ObjectId
from common.db import get_task_collection

# 全局任务停止标志 {task_id_str: bool}
_stop_flags = {}


class TaskManager:
    @staticmethod
    def generate_task_id():
        return str(int(time.time() * 1000000)) + str(int(time.time()) % 10)

    @staticmethod
    def create_task(params: dict) -> str:
        """
        初始化任务
        兼容接收 params 中的 '_id' 或 'task_id'
        """
        # 优先查找 _id，其次 task_id
        task_id = params.get("_id") or params.get("task_id")

        if not task_id:
            task_id = TaskManager.generate_task_id()

        # 确保 task_id 是字符串形式用于内存标记
        task_id_str = str(task_id)

        # 初始化时，确保没有停止标记
        if task_id_str in _stop_flags:
            del _stop_flags[task_id_str]

        # 尝试转换为 ObjectId 用于数据库查询
        try:
            oid = ObjectId(task_id)
        except Exception:
            oid = task_id  # 如果不是 ObjectId 格式，则保持原样

        doc_update = {
            "task_id": task_id_str,  # 保留一份字符串 ID
            "duration_sec": None,
            "ended_at": None,
            "error": None,
            "line_count": 0,
            "logs": [],
            "params": params,
            "started_at": datetime.utcnow(),
            "status": "running"
        }

        # 使用 update_one + upsert，基于 _id 更新
        get_task_collection().update_one(
            {"_id": oid},
            {"$set": doc_update},
            upsert=True
        )

        return task_id_str

    @staticmethod
    def mark_stop_requested(task_id: str):
        """API 调用此方法来请求停止任务"""
        if task_id:
            _stop_flags[str(task_id)] = True

    @staticmethod
    def should_stop(task_id: str) -> bool:
        """爬虫内部调用此方法检查是否应该停止"""
        if not task_id: return False
        return _stop_flags.get(str(task_id), False)

    @staticmethod
    def finish_task(task_id: str, status="stop", error=None):
        """
        任务结束
        """
        col = get_task_collection()
        task_id_str = str(task_id)

        try:
            oid = ObjectId(task_id)
        except Exception:
            oid = task_id

        # 清理停止标记
        if task_id_str in _stop_flags:
            del _stop_flags[task_id_str]

        # 查找任务以计算耗时
        task = col.find_one({"_id": oid})
        if not task:
            return

        end_time = datetime.utcnow()
        start_time = task.get("started_at", end_time)
        duration = (end_time - start_time).total_seconds()

        update_doc = {
            "status": status,
            "ended_at": end_time,
            "duration_sec": duration,
        }
        if error:
            update_doc["error"] = str(error)

        col.update_one({"_id": oid}, {"$set": update_doc})


class MongoTaskHandler(logging.Handler):
    def __init__(self, task_id):
        super().__init__()
        self.task_id = str(task_id)

        try:
            self.oid = ObjectId(task_id)
        except Exception:
            self.oid = task_id

        self.collection = get_task_collection()
        self.buffer = []
        self.last_flush = time.time()

    def emit(self, record):
        try:
            msg = self.format(record)
            log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {record.levelname}: {msg}"
            self.buffer.append(log_entry)
            if len(self.buffer) >= 10 or (time.time() - self.last_flush > 3):
                self.flush()
        except Exception:
            self.handleError(record)

    def flush(self):
        if not self.buffer:
            return
        try:
            # 基于 _id 写入日志
            result = self.collection.update_one(
                {"_id": self.oid},
                {
                    "$push": {"logs": {"$each": self.buffer}},
                    "$inc": {"line_count": len(self.buffer)}
                }
            )

            if result.matched_count == 0:
                print(f"[MongoTaskHandler Warning] No document matches _id: {self.oid}")

            self.buffer = []
            self.last_flush = time.time()

        except pymongo.errors.WriteError as e:
            # 【关键修复】捕获 "Cannot apply $inc to a value of non-numeric type" (Code 14)
            # 这意味着数据库里 line_count 是 null，我们需要手动修复它
            if e.code == 14 or "non-numeric type" in str(e):
                try:
                    # 1. 强制将 line_count 初始化为 0
                    self.collection.update_one(
                        {"_id": self.oid},
                        {"$set": {"line_count": 0}}
                    )
                    # 2. 重试写入日志
                    self.collection.update_one(
                        {"_id": self.oid},
                        {
                            "$push": {"logs": {"$each": self.buffer}},
                            "$inc": {"line_count": len(self.buffer)}
                        }
                    )
                    self.buffer = []
                    self.last_flush = time.time()
                except Exception as retry_e:
                    print(f"[MongoTaskHandler Retry Failed] {retry_e}")
            else:
                # 其他写入错误则打印
                print(f"[MongoTaskHandler WriteError] {e}")

        except Exception as e:
            print(f"[MongoTaskHandler Exception] Failed to write logs: {e}")
            traceback.print_exc()