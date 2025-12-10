import asyncio
from .crawler import run

# 【修改】增加 task_id 参数
async def launch(keywords: list[str], days: float, interval_min: int, task_id: str) -> None:
    loop = asyncio.get_event_loop()
    # 将 task_id 透传给 crawler.run
    await loop.run_in_executor(None, run, keywords, days, interval_min, task_id)