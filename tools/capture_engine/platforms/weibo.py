import asyncio
import time
import logging
import traceback
from datetime import datetime, timedelta

from config.settings import settings
from common.task_manager import TaskManager, MongoTaskHandler
from common.utils import get_logger
from data_collect.platforms.weibo.client import WeiboClient
from data_collect.platforms.weibo.storage import WeiboMongoStore

logger = get_logger(__name__)


async def run_task(keywords: list[str], days: float, interval_min: int, task_id: str = None):
    mongo_handler = None
    wb_logger = logging.getLogger("data_collect.platforms.weibo")

    if task_id:
        mongo_handler = MongoTaskHandler(task_id)
        logger.addHandler(mongo_handler)
        wb_logger.addHandler(mongo_handler)
        logger.info(f"[Weibo] Task started. TaskID={task_id}")
    else:
        logger.info(f"[Weibo] Direct Run. Keywords={keywords}")

    client = WeiboClient()
    store = WeiboMongoStore()
    deadline = datetime.now() + timedelta(days=days)

    try:
        while datetime.now() < deadline:
            # 检查停止信号
            if task_id and TaskManager.should_stop(task_id):
                logger.info("[Weibo] Stop signal detected.")
                break

            for kw in keywords:
                if task_id and TaskManager.should_stop(task_id): break

                total_saved = 0
                max_pages = (settings.CRAWL_LIMIT // 10) + 2

                logger.info(f"[Weibo] Searching: {kw}")
                for page in range(1, max_pages + 1):
                    if task_id and TaskManager.should_stop(task_id):
                        break

                    try:
                        posts = await client.search_posts(kw, page)
                        if not posts:
                            logger.info(f"[Weibo] No post found. kw={kw!r}, page={page}")
                            break

                        for post in posts:
                            if total_saved >= settings.CRAWL_LIMIT:
                                break
                            await store.save_post(post, [])
                            total_saved += 1
                            logger.info(f"   [{total_saved}] Saved: {post.note_id}")

                            # 新：抓取评论
                            try:
                                max_comments = getattr(settings, "WEIBO_COMMENT_LIMIT", 20)
                                comments = await client.get_comments(
                                    post.note_id,
                                    max_count=max_comments
                                )
                            except Exception as e:
                                logger.warning(f"[Weibo] Fetch comments failed for {post.note_id}: {e}")
                                comments = []

                            await store.save_post(post, comments)
                            total_saved += 1
                            logger.info(f"   [{total_saved}] Saved: {post.note_id}")

                    except Exception as e:
                        logger.warning(f"[Weibo] Page error: {e}")
                        break

            if not task_id and days < 0.1: break
            if datetime.now() >= deadline: break
            if task_id and TaskManager.should_stop(task_id): break

            await asyncio.sleep(interval_min * 60)

    except Exception as e:
        logger.error(f"[Weibo] Failed: {e}")
    finally:
        await client.close()
        if mongo_handler:
            mongo_handler.flush()
            logger.removeHandler(mongo_handler)
            wb_logger.removeHandler(mongo_handler)