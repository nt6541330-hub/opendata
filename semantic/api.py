from fastapi import APIRouter
import asyncio
from common.utils import get_logger
from semantic.pipeline import pipeline_instance

logger = get_logger(__name__)
router = APIRouter()

# 保持任务引用
_semantic_task = None


@router.post("/start_monitor", summary="启动语义处理自动监控")
async def start_monitor(interval: int = 60):
    global _semantic_task
    if pipeline_instance.running:
        return {"code": 400, "msg": "监控已经在运行中"}

    _semantic_task = asyncio.create_task(pipeline_instance.run_loop(interval))
    return {"code": 200, "msg": "语义处理监控已启动", "interval": interval}


@router.post("/stop_monitor", summary="停止语义处理自动监控")
async def stop_monitor():
    if not pipeline_instance.running:
        return {"code": 400, "msg": "监控未运行"}

    pipeline_instance.stop()
    return {"code": 200, "msg": "停止信号已发送，等待当前轮次结束"}


@router.post("/run_once", summary="手动触发一次处理流程")
async def run_once_manual():
    if pipeline_instance.running:
        return {"code": 400, "msg": "监控正在运行中，请勿手动触发"}

    try:
        # 异步调用同步方法
        result = await asyncio.to_thread(pipeline_instance.run_once)
        return {"code": 200, "msg": "执行完成", "detail": result}
    except Exception as e:
        return {"code": 500, "msg": f"执行失败: {str(e)}"}