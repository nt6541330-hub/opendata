import asyncio
from .crawler import run

# 增加 task_id 参数
async def launch(keywords: list[str], days: float, interval_min: int, task_id: str) -> None:
    loop = asyncio.get_event_loop()
    # 将 task_id 透传给同步的 run 函数
    await loop.run_in_executor(None, run, keywords, days, interval_min, task_id)