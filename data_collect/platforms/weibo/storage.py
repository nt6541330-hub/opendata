import asyncio
import os
import logging
from datetime import datetime
from urllib.parse import urlparse
from mimetypes import guess_type
from typing import List, Optional

import httpx
from bson import ObjectId
from pymongo import ASCENDING

from config.settings import settings
from common.db import get_weibo_storage
from common.utils import get_logger
from data_collect.utils.html import strip_html
from .field import WeiboPost, WeiboComment
from datetime import datetime
from data_collect.utils.html import strip_html

# 保留全局 logger 仅作为默认值
_global_logger = get_logger(__name__)


class WeiboMongoStore:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        初始化
        :param logger: 传入的任务专属 logger，如果未传则使用全局 logger
        """
        self.coll, self.fs = get_weibo_storage()
        self.coll.create_index([("note_id", ASCENDING)], unique=True)
        self.proxy = settings.HTTP_PROXY

        # 使用传入的 logger
        self.logger = logger if logger else _global_logger

    async def _download_image(self, url: str) -> Optional[bytes]:
        if not url: return None
        try:
            # 【修复点】适配新版 httpx
            proxy_url = self.proxy
            if proxy_url and not proxy_url.startswith("http"):
                proxy_url = f"http://{proxy_url}"

            # 使用 proxy (单数)
            async with httpx.AsyncClient(timeout=20, proxy=proxy_url, verify=False) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp.content
        except Exception as e:
            self.logger.warning(f"[mongo] download image error url={url} err={e}")
        return None

    async def _save_image_to_gridfs(self, data: bytes, url: str) -> ObjectId:
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path) or "weibo_img.jpg"
        content_type, _ = guess_type(filename)
        if not content_type:
            content_type = "image/jpeg"

        def _put() -> ObjectId:
            return self.fs.put(
                data,
                filename=filename,
                source_url=url,
                content_type=content_type,
            )

        file_id: ObjectId = await asyncio.to_thread(_put)
        return file_id

    async def _handle_images(self, post: WeiboPost) -> List[ObjectId]:
        image_ids: List[ObjectId] = []
        if not post.images: return image_ids

        for img in post.images:
            if not img.url: continue
            data = await self._download_image(img.url)
            if not data: continue
            try:
                file_id = await self._save_image_to_gridfs(data, img.url)
                image_ids.append(file_id)
            except Exception:
                pass
        return image_ids

    async def save_post(self, post: WeiboPost, comments: List[WeiboComment]) -> None:
        if not post.note_id:
            return

        post_text = strip_html(post.text_html)
        comments_text = [strip_html(c.text_html) for c in comments if c.text_html]
        image_ids = await self._handle_images(post)

        created_at = post.raw.get("created_at", "")
        try:
            pub_time = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
            pub_time = pub_time.replace(tzinfo=None)
        except Exception:
            pub_time = datetime.utcnow()

        # 新增：构造标题
        title = ""
        raw = post.raw or {}
        # 有些 m.weibo.cn 返回里可能带有类似 title/status_title/page_title 的字段
        for key in ("title", "status_title", "page_title"):
            if raw.get(key):
                title = strip_html(str(raw.get(key)))
                break
        if not title:
            # 退化方案：用正文第一行的前 50 个字符作为标题
            title = post_text.split("\n", 1)[0][:50]

        doc = {
            "note_id": post.note_id,
            "keyword": post.keyword,
            "title": title,  # <-- 新字段
            "content": post_text,
            "comments": comments_text,
            "images": image_ids,
            "time": pub_time,
            "crawl_time": datetime.utcnow(),
            "source": "微博",
            "status": "0"
        }

        self.coll.update_one({"note_id": post.note_id}, {"$set": doc}, upsert=True)
        self.logger.info(f"[Weibo] Saved: {post.note_id} Time: {pub_time}")