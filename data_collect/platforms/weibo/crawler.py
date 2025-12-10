# open_source_data/data_collect/platforms/weibo/crawler.py
from typing import List
import asyncio
from config.settings import settings
from common.utils import get_logger
from common.task_manager import TaskManager
from .client import WeiboClient
from .storage import WeiboMongoStore
from .exception import CookieInvalidError

logger = get_logger(__name__)


class WeiboCrawler:
    def __init__(self) -> None:
        self.client = WeiboClient()
        self.store = WeiboMongoStore()

    async def run(self, keywords: List[str], task_id: str = None) -> None:
        """
        执行一轮关键词采集
        """
        limit = settings.CRAWL_LIMIT

        for kw in keywords:
            if task_id and TaskManager.should_stop(task_id): break

            kw = kw.strip()
            if not kw: continue

            logger.info(f"[Weibo] Searching: {kw}")
            total_saved = 0
            page = 1

            while total_saved < limit:
                if task_id and TaskManager.should_stop(task_id): break

                try:
                    posts = await self.client.search_posts(kw, page)
                    if not posts: break

                    for post in posts:
                        if total_saved >= limit:
                            break

                        # 新：抓取评论（默认最多 20 条，可用配置控制）
                        try:
                            max_comments = getattr(settings, "WEIBO_COMMENT_LIMIT", 20)
                            comments = await self.client.get_comments(
                                post.note_id,
                                max_count=max_comments
                            )
                        except Exception as e:
                            logger.warning(f"[Weibo] Failed to fetch comments for {post.note_id}: {e}")
                            comments = []

                        await self.store.save_post(post, comments)
                        total_saved += 1

                    page += 1
                    await asyncio.sleep(2)  # 稍微停顿防止封号

                except CookieInvalidError:
                    logger.error("[Weibo] Cookie Invalid! Stop crawling.")
                    return
                except Exception as e:
                    logger.warning(f"[Weibo] Error on page {page}: {e}")
                    break

            logger.info(f"[Weibo] Finished {kw}: {total_saved}")

    async def close(self):
        await self.client.close()