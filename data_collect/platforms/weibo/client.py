import json
import os
from typing import Any, Dict, List, Optional
import httpx
import logging

from config.settings import settings
from common.utils import get_logger
from .exception import CookieInvalidError, ApiError
from .field import WeiboPost, WeiboImage, WeiboComment

# 保留全局 logger 仅作为默认值，或者用于非实例方法的日志
_global_logger = get_logger(__name__)


class WeiboClient:
    """
    负责调用 m.weibo.cn 的搜索和评论接口
    """
    BASE_M_API = "https://m.weibo.cn/api"

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        初始化
        :param logger: 传入的任务专属 logger，如果未传则使用全局 logger
        """
        self.timeout = 15
        self.proxy = settings.HTTP_PROXY
        self._client: Optional[httpx.AsyncClient] = None

        # 使用传入的 logger，如果没有则使用全局的
        self.logger = logger if logger else _global_logger

        self._cookie_header = self._load_cookie_header()

    def _load_cookie_header(self) -> str:
        cookie_file = settings.WEIBO_COOKIE_FILE
        if not os.path.exists(cookie_file):
            self.logger.warning(f"[Weibo] Cookie file not found: {cookie_file}")
            return ""

        try:
            with open(cookie_file, "r", encoding="utf-8") as f:
                cookies = json.load(f)

            parts = []
            for c in cookies:
                name = c.get("name")
                value = c.get("value")
                if name and value:
                    parts.append(f"{name}={value}")

            if not parts:
                return ""
            return "; ".join(parts)
        except Exception as e:
            self.logger.error(f"[Weibo] Failed to load cookie: {e}")
            return ""

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {
                "Accept": "application/json, text/plain, */*",
                "MWeibo-Pwa": "1",
                "User-Agent": "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.91 Mobile Safari/537.36",
                "X-Requested-With": "XMLHttpRequest"
            }
            if self._cookie_header:
                headers["Cookie"] = self._cookie_header

            # 【关键修复】适配新版 httpx，使用 proxy (单数)
            proxy_url = self.proxy
            if proxy_url and not proxy_url.startswith("http"):
                proxy_url = f"http://{proxy_url}"

            # 注意：新版 httpx 若不传 proxy 参数则设为 None（不要传空字典）
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                proxy=proxy_url,  # <--- 必须是 proxy，不是 proxies
                headers=headers,
                follow_redirects=True,
                verify=False
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _ensure_json_or_raise(resp: httpx.Response) -> None:
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            return

        text = resp.text[:500]
        risky = ["登录", "安全验证", "验证码", "抱歉，出错啦"]
        if any(k in text for k in risky):
            raise CookieInvalidError("微博返回登录/风控页面，Cookie 可能已失效")

    async def search_posts(self, keyword: str, page: int = 1) -> List[WeiboPost]:
        client = await self._get_client()
        url = f"{self.BASE_M_API}/container/getIndex"
        params = {
            "containerid": f"100103type=1&q={keyword}",
            "page_type": "searchall",
            "page": page
        }

        try:
            resp = await client.get(
                url,
                params=params,
                headers={"Referer": "https://m.weibo.cn/search"}
            )
            if resp.status_code != 200:
                raise ApiError(f"HTTP {resp.status_code}")

            self._ensure_json_or_raise(resp)
            data = resp.json()
        except Exception as e:
            # 抛出去，让上层打印 [Weibo] Page error
            raise e

        cards = data.get("data", {}).get("cards", []) or []
        posts: list[WeiboPost] = []

        def _build_post(mblog: dict) -> WeiboPost | None:
            if not mblog:
                return None
            note_id = str(mblog.get("id") or mblog.get("mid") or "")
            if not note_id:
                return None

            text_html = mblog.get("text", "") or ""
            pics = mblog.get("pics", []) or []
            images: list[WeiboImage] = []
            for p in pics:
                u = p.get("large", {}).get("url") or p.get("url")
                if u:
                    images.append(WeiboImage(url=u))

            return WeiboPost(
                note_id=note_id,
                keyword=keyword,
                text_html=text_html,
                images=images,
                raw=mblog
            )

        # 第一层：cards 里直接带 mblog 的（常见）
        for card in cards:
            ctype = card.get("card_type")
            if ctype == 9 and card.get("mblog"):
                post = _build_post(card.get("mblog"))
                if post:
                    posts.append(post)

            # 第二层：话题 / 推荐卡片里嵌套的 card_group
            card_group = card.get("card_group") or []
            for sub in card_group:
                if sub.get("card_type") == 9 and sub.get("mblog"):
                    post = _build_post(sub.get("mblog"))
                    if post:
                        posts.append(post)

        if not posts:
            # 增加一点调试信息，方便排查是结构问题还是风控问题
            self.logger.warning(
                f"[Weibo] No posts parsed. keyword={keyword!r}, page={page}, "
                f"ok={data.get('ok')}, cards={len(cards)}"
            )

        return posts

    async def get_comments(self, note_id: str, max_count: int = 20) -> List[WeiboComment]:
        client = await self._get_client()
        url = f"{self.BASE_M_API}/comments/show"
        params = {"id": note_id, "mid": note_id}

        try:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                data_list = data.get("data", {}).get("data", [])
                comments = []
                for c in data_list:
                    comments.append(WeiboComment(
                        cid=str(c.get("id")),
                        text_html=c.get("text", ""),
                        raw=c
                    ))
                return comments[:max_count]
        except:
            pass
        return []