from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional
import asyncio
import traceback

from common.utils import get_logger
from common.task_manager import TaskManager

# 引入各平台爬虫的启动函数
from .platforms.toutiao import run_task as run_toutiao
from .platforms.weibo import run_task as run_weibo
from .platforms.cctv import run_task as run_cctv
from .platforms.xinhua import run_task as run_xinhua
from .platforms.cnn import run_task as run_cnn

logger = get_logger(__name__)
router = APIRouter()


class CaptureRequest(BaseModel):
    # 【修改】使用 alias="_id" 接收前端传来的 _id 参数，代码中用 self.id 访问
    id: str = Field(..., alias="_id")
    keywords: str  # 关键词
    start_time: str  # 开始时间
    end_time: str  # 结束时间
    interval_minutes: int = 60


class StopRequest(BaseModel):
    # 【新增】停止接口也接收 _id
    id: str = Field(..., alias="_id")


@router.post("/run", summary="启动定时抓取任务")
async def run_capture_task(req: CaptureRequest):
    kw_list = req.keywords.strip().split()
    if not kw_list:
        return {"code": 400, "msg": "关键词不能为空"}

    try:
        start_dt = datetime.strptime(req.start_time, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(req.end_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"code": 400, "msg": "时间格式错误"}

    now = datetime.now()
    if end_dt <= start_dt: return {"code": 400, "msg": "结束时间错误"}
    if end_dt <= now: return {"code": 400, "msg": "结束时间已过"}

    # 1. 创建/更新任务记录
    is_future = start_dt > now

    # 使用 by_alias=True 确保生成的 dict 包含 "_id" 键
    task_params = req.dict(by_alias=True)
    task_params.update({
        "platform": "all_5_platforms",
        "parsed_keywords": kw_list,
        "status_desc": "Scheduled" if is_future else "Running"
    })

    # 获取任务ID (即 _id)
    task_id = req.id

    try:
        # TaskManager 内部会识别 _id 并更新数据库
        TaskManager.create_task(task_params)
        logger.info(f"[CaptureEngine] Task Initialized: {task_id}")
    except Exception as e:
        logger.warning(f"Task creation warning: {e}")

    # 2. 启动调度器
    run_days = (end_dt - start_dt).total_seconds() / 86400.0
    if run_days < 0: run_days = 0.01

    wait_seconds = (start_dt - now).total_seconds()

    if wait_seconds > 0:
        asyncio.create_task(
            _delayed_scheduler(wait_seconds, kw_list, run_days, req.interval_minutes, task_id, end_dt))
        msg = f"任务已预约，将于 {req.start_time} 启动"
    else:
        # 立即启动调度器
        asyncio.create_task(_scheduler(kw_list, run_days, req.interval_minutes, task_id, end_dt))
        msg = "任务立即启动"

    return {"code": 200, "msg": msg, "task_id": task_id}


@router.post("/stop", summary="停止任务")
async def stop_task(req: StopRequest):
    # 【修改】接收 StopRequest 对象，使用 req.id
    task_id = req.id
    TaskManager.mark_stop_requested(task_id)
    logger.info(f"[CaptureEngine] Manual STOP signal received for {task_id}")
    return {"code": 200, "msg": "停止信号已发送"}


async def _delayed_scheduler(delay, keywords, days, interval, task_id, end_dt):
    try:
        logger.info(f"[CaptureEngine] Task {task_id} waiting {delay:.1f}s...")
        start_wait = datetime.now()
        while (datetime.now() - start_wait).total_seconds() < delay:
            if TaskManager.should_stop(task_id):
                TaskManager.finish_task(task_id, status="stop")
                return
            await asyncio.sleep(1)

        await _scheduler(keywords, days, interval, task_id, end_dt)
    except Exception as e:
        TaskManager.finish_task(task_id, status="failed", error=str(e))


async def _scheduler(keywords, days, interval, task_id, end_dt):
    """
    【核心调度器】
    """
    logger.info(f"[CaptureEngine] Scheduler started for {task_id}")

    # 启动所有子任务，传递 task_id 用于日志和控制
    crawlers = [
        asyncio.create_task(run_toutiao(keywords, days, interval, task_id), name="toutiao"),
        asyncio.create_task(run_weibo(keywords, days, interval, task_id), name="weibo"),
        asyncio.create_task(run_cctv(keywords, days, interval, task_id), name="cctv"),
        asyncio.create_task(run_xinhua(keywords, days, interval, task_id), name="xinhua"),
        asyncio.create_task(run_cnn(keywords, days, interval, task_id), name="cnn"),
    ]

    final_status = "finished"
    error_msg = None

    try:
        while True:
            # A. 检查自然结束
            if all(t.done() for t in crawlers):
                logger.info(f"[CaptureEngine] All crawlers finished naturally.")
                break

            # B. 检查手动停止
            if TaskManager.should_stop(task_id):
                logger.info(f"[CaptureEngine] Manual STOP detected for {task_id}.")
                final_status = "stop"
                break

            # C. 检查时间结束
            if datetime.now() >= end_dt:
                logger.info(f"[CaptureEngine] Time is up for {task_id}.")
                TaskManager.mark_stop_requested(task_id)
                final_status = "stop"
                break

            await asyncio.sleep(2)

        # 等待退出
        done, pending = await asyncio.wait(crawlers, timeout=30)
        if pending:
            logger.warning(f"[CaptureEngine] Some crawlers did not exit in time, forcing cancel.")
            for t in pending: t.cancel()

    except Exception as e:
        logger.error(f"[CaptureEngine] Scheduler error: {e}")
        final_status = "failed"
        error_msg = str(e)

    # 更新数据库
    TaskManager.finish_task(task_id, status=final_status, error=error_msg)
    logger.info(f"[CaptureEngine] Task {task_id} lifecycle ended. Status: {final_status}")