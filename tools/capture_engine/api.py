from fastapi import APIRouter, Body, Request
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional
import asyncio
import traceback
import json

from common.utils import get_logger
from common.task_manager import TaskManager, MongoTaskHandler

# 引入各平台爬虫的启动函数
from .platforms.toutiao import run_task as run_toutiao
from .platforms.weibo import run_task as run_weibo
from .platforms.cctv import run_task as run_cctv
from .platforms.xinhua import run_task as run_xinhua
from .platforms.cnn import run_task as run_cnn

# 保留全局 logger 用于 API 接口层面的系统日志（非任务日志）
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
async def run_capture_task(req: CaptureRequest, request: Request):
    # =========================================================================
    # 【新增调试代码】打印平台发送过来的原始数据
    # =========================================================================
    try:
        raw_body = await request.json()
        logger.info(f"\n{'=' * 20} [DEBUG: Platform Request] {'=' * 20}")
        logger.info(f"Raw JSON Received: {json.dumps(raw_body, ensure_ascii=False)}")
        logger.info(f"Parsed ID (req.id): '{req.id}'")
        logger.info(f"{'=' * 60}\n")
    except Exception as e:
        logger.error(f"[DEBUG] Failed to read raw body: {e}")
    # =========================================================================

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

    # 获取初始 ID (可能是空字符串)
    initial_id = req.id
    task_id = initial_id

    try:
        # 【关键修复】接收 create_task 返回的真实 ID
        # 如果 initial_id 为空，TaskManager 会生成新 ID 并返回
        real_task_id = TaskManager.create_task(task_params)
        logger.info(f"[CaptureEngine] Task Initialized. ReqID='{initial_id}', RealID='{real_task_id}'")

        # 将 task_id 更新为真实 ID，确保后续调度和日志使用正确的 ID
        task_id = real_task_id

        if initial_id != real_task_id:
            logger.warning(f"⚠️ [Alert] ID Mismatch! Platform sent: '{initial_id}', DB generated: '{real_task_id}'")

    except Exception as e:
        logger.warning(f"Task creation warning: {e}")
        # 如果出错且没有 ID，生成一个临时 fallback ID 防止报错
        if not task_id:
            task_id = f"fallback_{int(now.timestamp())}"

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

    # 返回真实的 task_id
    return {"code": 200, "msg": msg, "task_id": task_id}


@router.post("/stop", summary="停止任务")
async def stop_task(req: StopRequest):
    # 【修改】接收 StopRequest 对象，使用 req.id
    task_id = req.id
    TaskManager.mark_stop_requested(task_id)
    logger.info(f"[CaptureEngine] Manual STOP signal received for {task_id}")
    return {"code": 200, "msg": "停止信号已发送"}


async def _delayed_scheduler(delay, keywords, days, interval, task_id, end_dt):
    # 为延迟调度器创建临时 logger，以便记录“等待中”的状态
    task_logger = get_logger(f"{__name__}.scheduler.{task_id}")
    mongo_handler = MongoTaskHandler(task_id)
    task_logger.addHandler(mongo_handler)

    try:
        task_logger.info(f"[CaptureEngine] Task {task_id} waiting {delay:.1f}s...")
        start_wait = datetime.now()
        while (datetime.now() - start_wait).total_seconds() < delay:
            if TaskManager.should_stop(task_id):
                TaskManager.finish_task(task_id, status="stop")
                task_logger.info(f"[CaptureEngine] Task stopped during wait period.")
                return
            await asyncio.sleep(1)

        # 移除 handler，因为 _scheduler 内部会重新建立它自己的 handler
        mongo_handler.flush()
        task_logger.removeHandler(mongo_handler)

        await _scheduler(keywords, days, interval, task_id, end_dt)
    except Exception as e:
        task_logger.error(f"Delayed scheduler error: {e}")
        TaskManager.finish_task(task_id, status="failed", error=str(e))
    finally:
        # 确保清理
        if mongo_handler in task_logger.handlers:
            mongo_handler.flush()
            task_logger.removeHandler(mongo_handler)


async def _scheduler(keywords, days, interval, task_id, end_dt):
    """
    【核心调度器】
    """
    # =========================================================================
    # 【核心修复】创建调度器专属 Logger，确保调度日志（开始/结束）入库
    # =========================================================================
    task_logger = get_logger(f"{__name__}.scheduler.{task_id}")
    mongo_handler = MongoTaskHandler(task_id)
    task_logger.addHandler(mongo_handler)

    task_logger.info(f"[CaptureEngine] Scheduler started for {task_id}")

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
                task_logger.info(f"[CaptureEngine] All crawlers finished naturally.")
                break

            # B. 检查手动停止
            if TaskManager.should_stop(task_id):
                task_logger.info(f"[CaptureEngine] Manual STOP detected for {task_id}.")
                final_status = "stop"
                break

            # C. 检查时间结束
            if datetime.now() >= end_dt:
                task_logger.info(f"[CaptureEngine] Time is up for {task_id}.")
                TaskManager.mark_stop_requested(task_id)
                final_status = "stop"
                break

            await asyncio.sleep(2)

        # 等待退出
        done, pending = await asyncio.wait(crawlers, timeout=30)
        if pending:
            task_logger.warning(f"[CaptureEngine] Some crawlers did not exit in time, forcing cancel.")
            for t in pending: t.cancel()

    except Exception as e:
        task_logger.error(f"[CaptureEngine] Scheduler error: {e}")
        task_logger.error(traceback.format_exc())
        final_status = "failed"
        error_msg = str(e)

    # 更新数据库
    TaskManager.finish_task(task_id, status=final_status, error=error_msg)
    task_logger.info(f"[CaptureEngine] Task {task_id} lifecycle ended. Status: {final_status}")

    # 【关键】清理 Handler
    mongo_handler.flush()
    task_logger.removeHandler(mongo_handler)