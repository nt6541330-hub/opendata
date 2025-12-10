import time
import random
import os
import json
import requests
import io
import traceback
from datetime import datetime, timedelta
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from DrissionPage import ChromiumPage, ChromiumOptions
from config.settings import settings
from common.db import get_toutiao_storage
from common.utils import get_logger
from common.task_manager import TaskManager, MongoTaskHandler

logger = get_logger(__name__)
coll, grid_fs = get_toutiao_storage()

MOBILE_UA = "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.91 Mobile Safari/537.36"
MAX_ITEMS_PER_KEYWORD = 40
img_executor = ThreadPoolExecutor(max_workers=10)


def download_one_image(url):
    if not url or url.startswith("data:"): return None
    try:
        proxies = {"http": settings.HTTP_PROXY, "https": settings.HTTP_PROXY} if settings.HTTP_PROXY else None
        resp = requests.get(url, timeout=3, headers={"User-Agent": MOBILE_UA}, proxies=proxies)
        if resp.status_code == 200:
            return grid_fs.put(io.BytesIO(resp.content), filename=url.split("/")[-1][:50])
    except:
        pass
    return None


def load_cookies(page: ChromiumPage):
    cookie_file = settings.TOUTIAO_COOKIE_FILE
    if not os.path.exists(cookie_file):
        logger.warning(f"[Toutiao] No cookie file found.")
        return
    try:
        with open(cookie_file, 'r', encoding='utf-8') as f:
            cookies = json.load(f)
        page.get("https://m.toutiao.com/")
        page.set.cookies(cookies)
        logger.info(f"[Toutiao] Cookies loaded.")
        time.sleep(1)
    except Exception:
        pass


def get_mobile_detail_reuse(tab, url: str):
    full_text = ""
    image_ids = []
    pub_time = None
    try:
        if not tab.get(url, timeout=5): return None, [], None
        ele = tab.ele('tag:article', timeout=2) or tab.ele('css:.article-content', timeout=1) or tab.ele(
            'css:.tt-content', timeout=1)

        # 尝试提取时间
        try:
            time_ele = tab.ele('css:.article-meta', timeout=0.1) or tab.ele('css:.publish-time', timeout=0.1)
            if time_ele:
                # 简单处理时间文本
                time_text = time_ele.text
                # 这里可以接入更复杂的正则解析，暂用当前时间兜底
                pass
        except:
            pass

        if ele:
            full_text = ele.text.strip()
            img_urls = [img.attr('src') for img in ele.eles('tag:img') if img.attr('src')]
            if img_urls:
                futures = [img_executor.submit(download_one_image, u) for u in img_urls]
                for f in as_completed(futures):
                    if fid := f.result(): image_ids.append(fid)
    except:
        pass
    return full_text, image_ids, pub_time


def _run_sync(keywords: list, days: float, interval: int, task_id: str):
    deadline = datetime.now() + timedelta(days=days)

    mongo_handler = None
    if task_id:
        mongo_handler = MongoTaskHandler(task_id)
        logger.addHandler(mongo_handler)
        logger.info(f"[Toutiao] Task started. TaskID={task_id}")
    else:
        logger.info(f"[Toutiao] Direct Run. Keywords={keywords}")

    co = ChromiumOptions()
    co.headless(True)
    co.set_argument('--no-sandbox')
    co.set_argument('--disable-gpu')
    co.set_user_agent(MOBILE_UA)
    co.auto_port()
    if settings.HTTP_PROXY: co.set_proxy(settings.HTTP_PROXY)

    try:
        page = ChromiumPage(co)
        load_cookies(page)
        detail_tab = page.new_tab()

        while datetime.now() < deadline:
            # 停止信号检查
            if task_id and TaskManager.should_stop(task_id):
                logger.info("[Toutiao] Stop signal received.")
                break

            for kw in keywords:
                if task_id and TaskManager.should_stop(task_id): break
                _crawl_keyword(page, detail_tab, kw)
                time.sleep(random.uniform(2, 4))

            if not task_id and days < 0.1: break
            if datetime.now() >= deadline: break
            if task_id and TaskManager.should_stop(task_id): break

            time.sleep(interval * 60)

    except Exception as e:
        logger.error(f"[Toutiao] Error: {e}")
    finally:
        try:
            page.quit()
        except:
            pass
        img_executor.shutdown(wait=False)
        if mongo_handler:
            mongo_handler.flush()
            logger.removeHandler(mongo_handler)


def _crawl_keyword(page, detail_tab, keyword):
    logger.info(f"[Toutiao] Searching: {keyword}")
    url = f"https://so.toutiao.com/search?keyword={quote(keyword)}&pd=synthesis&source=search_history"
    page.get(url)
    time.sleep(3)

    if "验证" in page.title: return
    try:
        page.ele('text=资讯', timeout=2).click()
    except:
        pass

    total = 0
    seen = set()
    for i in range(10):
        if total >= MAX_ITEMS_PER_KEYWORD: break
        page.scroll.down(1500)
        time.sleep(2)
        for a in page.eles('tag:a'):
            if total >= MAX_ITEMS_PER_KEYWORD: break
            try:
                link = a.link
                if not link or len(a.text) < 5: continue
                if link in seen: continue
                seen.add(link)
                if "toutiao.com" not in link and "snssdk.com" not in link: continue
                if any(x in link for x in ["/video/", "search?", "user/token"]): continue

                if coll.find_one({"url": link}): continue

                content, imgs, pub_time = get_mobile_detail_reuse(detail_tab, link)

                if content:
                    # 确保 time 字段存在
                    final_time = pub_time if pub_time else datetime.now()

                    coll.insert_one({
                        "title": a.text,
                        "url": link,
                        "content": content,
                        "images": imgs,
                        "source": "今日头条",
                        "keyword": keyword,
                        "time": final_time,  # 新闻发布时间
                        "crawl_time": datetime.now(),  # 抓取时间
                        "status": "0"  # 默认状态
                    })
                    total += 1
                    logger.info(f"   [{total}] Saved: {a.text[:10]}...")
            except:
                pass
    logger.info(f"[Toutiao] Finished {keyword}: {total}")


async def run_task(keywords, days, interval, task_id=None):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_sync, keywords, days, interval, task_id)