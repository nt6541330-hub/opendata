from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import logging
import traceback
import time

# 引入各平台的启动函数
from data_collect.platforms.weibo.crawler import WeiboCrawler
from data_collect.platforms.cctv.launch import launch as cctv_launch
from data_collect.platforms.toutiao.launch import launch as toutiao_launch
from data_collect.platforms.xinhua.launch import launch as xinhua_launch
from data_collect.platforms.cnn.launch import launch as cnn_launch

from common.utils import get_logger
from common.task_manager import TaskManager, MongoTaskHandler

logger = get_logger(__name__)
router = APIRouter()


# 通用请求体模型
class TaskModel(BaseModel):
    keywords: List[str]
    days: float = 0.5  # 持续运行天数
    interval_minutes: int = 60  # 轮次间隔
    task_id: Optional[str] = None  # 支持前端指定任务ID


# ===================== 微博 (Weibo) =====================
@router.post("/weibo")
async def run_weibo(task: TaskModel):
    """启动微博采集 (Async)"""
    # 1. 准备参数
    params = task.dict()
    params.update({
        "platform": "weibo",
        "news_collection": "source_weibo",
        "news_fs": "fs_weibo"
    })

    # 2. 创建任务
    task_id = TaskManager.create_task(params)
    logger.info(f"Task Created: {task_id} | Platform: Weibo | Keywords: {task.keywords}")

    # 3. 启动后台任务 (传入 task_id)
    asyncio.create_task(_weibo_bg(task.keywords, task.days, task.interval_minutes, task_id))

    return {
        "msg": "Weibo task started",
        "task_id": task_id,
        "params": params
    }


async def _weibo_bg(keywords, days, interval, task_id):
    """微博后台任务逻辑 (带日志入库)"""
    # 获取微博模块的 logger，以便截获其日志
    weibo_logger = logging.getLogger("data_collect.platforms.weibo")

    # 挂载 MongoDB 日志处理器
    mongo_handler = MongoTaskHandler(task_id)
    weibo_logger.addHandler(mongo_handler)
    logger.addHandler(mongo_handler)  # 同时也捕获当前函数的日志

    crawler = WeiboCrawler()
    end_time = time.time() + days * 86400

    try:
        logger.info(f"[Weibo] Task started. TaskID={task_id}")

        while time.time() < end_time:
            await crawler.run(keywords)

            if time.time() >= end_time:
                break

            logger.info(f"[Weibo] Round finished. Sleeping {interval} min...")
            await asyncio.sleep(interval * 60)

        # 任务正常结束
        TaskManager.finish_task(task_id, status="finished")
        logger.info("[Weibo] Task finished successfully.")

    except Exception as e:
        # 任务异常
        err_msg = f"{e}\n{traceback.format_exc()}"
        logger.error(f"[Weibo] Task Failed: {e}")
        TaskManager.finish_task(task_id, status="failed", error=err_msg)

    finally:
        # 清理资源
        await crawler.client.close()
        mongo_handler.flush()
        weibo_logger.removeHandler(mongo_handler)
        logger.removeHandler(mongo_handler)


# ===================== 央视 (CCTV) =====================
@router.post("/cctv")
async def run_cctv(task: TaskModel):
    params = task.dict()
    params.update({"platform": "cctv", "news_collection": "source_cctv"})

    task_id = TaskManager.create_task(params)
    logger.info(f"Task Created: {task_id} | Platform: CCTV")

    # 传递 task_id 给 launch
    asyncio.create_task(cctv_launch(task.keywords, task.days, task.interval_minutes, task_id))

    return {"msg": "CCTV task started", "task_id": task_id, "params": params}


# ===================== 今日头条 (Toutiao) =====================
@router.post("/toutiao")
async def run_toutiao(task: TaskModel):
    params = task.dict()
    params.update({
        "platform": "toutiao",
        "news_collection": "source_toutiao",
        "news_fs": "fs_toutiao"
    })

    task_id = TaskManager.create_task(params)
    logger.info(f"Task Created: {task_id} | Platform: Toutiao")

    asyncio.create_task(toutiao_launch(task.keywords, task.days, task.interval_minutes, task_id))

    return {"msg": "Toutiao task started", "task_id": task_id, "params": params}


# ===================== 新华网 (Xinhua) =====================
@router.post("/xinhua")
async def run_xinhua(task: TaskModel):
    params = task.dict()
    params.update({"platform": "xinhua", "news_collection": "source_xinHua_net"})

    task_id = TaskManager.create_task(params)
    logger.info(f"Task Created: {task_id} | Platform: Xinhua")

    asyncio.create_task(xinhua_launch(task.keywords, task.days, task.interval_minutes, task_id))

    return {"msg": "Xinhua task started", "task_id": task_id, "params": params}


# ===================== CNN =====================
@router.post("/cnn")
async def run_cnn(task: TaskModel):
    params = task.dict()
    params.update({"platform": "cnn", "news_collection": "source_cnn"})

    task_id = TaskManager.create_task(params)
    logger.info(f"Task Created: {task_id} | Platform: CNN")

    asyncio.create_task(cnn_launch(task.keywords, task.days, task.interval_minutes, task_id))

    return {"msg": "CNN task started", "task_id": task_id, "params": params}