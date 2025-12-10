import time
import traceback
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import asyncio

from config.settings import settings
from common.db import get_cnn_storage
from common.utils import get_logger
from common.task_manager import TaskManager, MongoTaskHandler

logger = get_logger(__name__)
collection, fs = get_cnn_storage()


def translate_text(text: str, target_lang="zh") -> str:
    # 这里可保留原有的 Langchain 逻辑，此处简化
    return text


def _run_sync(keywords, days, interval, task_id):
    end_time = time.time() + days * 86400
    mongo_handler = None
    if task_id:
        mongo_handler = MongoTaskHandler(task_id)
        logger.addHandler(mongo_handler)
        logger.info(f"[CNN] Task started. TaskID={task_id}")

    opt = webdriver.ChromeOptions()
    opt.add_argument("--headless=new")
    opt.add_argument("--no-sandbox")
    opt.add_argument("--disable-gpu")
    if settings.HTTP_PROXY: opt.add_argument(f"--proxy-server={settings.HTTP_PROXY}")

    driver = None
    try:
        driver = webdriver.Chrome(service=Service(settings.DRIVER_PATH), options=opt)
        while time.time() < end_time:
            if task_id and TaskManager.should_stop(task_id):
                logger.info("[CNN] Stop signal received.")
                break

            for kw in keywords:
                if task_id and TaskManager.should_stop(task_id): break
                _crawl_one(driver, kw, task_id)

            if time.time() >= end_time: break
            if task_id and TaskManager.should_stop(task_id): break
            time.sleep(interval * 60)

    except Exception as e:
        logger.error(f"[CNN] Error: {e}")
    finally:
        if driver: driver.quit()
        if mongo_handler:
            mongo_handler.flush()
            logger.removeHandler(mongo_handler)


def _crawl_one(driver, kw, task_id):
    logger.info(f"[CNN] Searching: {kw}")
    try:
        driver.get(f"https://edition.cnn.com/search?q={kw}&size=10&types=article")
        time.sleep(5)

        links = []
        try:
            elems = driver.find_elements(By.CSS_SELECTOR, "div.card a")
            links = [e.get_attribute("href") for e in elems if e.get_attribute("href")]
        except:
            pass

        count = 0
        for url in links:
            if count >= settings.CRAWL_LIMIT: break
            if task_id and TaskManager.should_stop(task_id): break
            if collection.find_one({"url": url}): continue

            try:
                driver.get(url)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
                title = driver.find_element(By.TAG_NAME, "h1").text
                content = ""
                try:
                    content = driver.find_element(By.TAG_NAME, "article").text
                except:
                    try:
                        content = driver.find_element(By.CSS_SELECTOR, ".article__content").text
                    except:
                        pass

                if not content: continue

                # 简单的时间提取
                pub_time = datetime.now()
                try:
                    # CNN 的时间通常在 .timestamp
                    ts = driver.find_element(By.CSS_SELECTOR, '.timestamp').text
                    # 此处略去复杂解析，保留占位
                except:
                    pass

                collection.insert_one({
                    "title": title,
                    "title_zh": title,  # 占位
                    "content": content,
                    "content_zh": content[:200],
                    "url": url,
                    "source": "CNN",
                    "keyword": kw,
                    "time": pub_time,  # 新闻发布时间
                    "crawl_time": datetime.now(),
                    "status": "0"  # 默认状态 0
                })
                count += 1
                logger.info(f"   [{count}] Saved: {title[:15]}...")
            except:
                pass
    except:
        pass


async def run_task(keywords, days, interval, task_id=None):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_sync, keywords, days, interval, task_id)