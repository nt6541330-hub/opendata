import time
import requests
import io
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from fake_useragent import UserAgent
import asyncio

from config.settings import settings
from common.db import get_xinhua_storage
from common.utils import get_logger
from common.task_manager import TaskManager, MongoTaskHandler

logger = get_logger(__name__)
col, fs = get_xinhua_storage()


def download_to_gridfs(url):
    try:
        if not url or url.startswith('data:'): return None
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            return fs.upload_from_stream(url.split('/')[-1], io.BytesIO(r.content))
    except:
        pass
    return None


def parse_xinhua_time(driver):
    try:
        src = driver.page_source
        m = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', src)
        if m: return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        m = re.search(r'(\d{4}-\d{2}-\d{2})', src)
        if m: return datetime.strptime(m.group(1), "%Y-%m-%d")
    except:
        pass
    return datetime.now()


def extract_detail(driver, kw):
    try:
        title = driver.find_element(By.CSS_SELECTOR, 'h1 span.title').text.strip()
        content = driver.find_element(By.ID, 'detail').text.strip()
        imgs = [img.get_attribute('src') for img in driver.find_elements(By.CSS_SELECTOR, 'img')]
        pub_time = parse_xinhua_time(driver)
        return {
            'title': title,
            'source': '新华网',
            'content': content,
            'images': imgs,
            'url': driver.current_url,
            'keyword': kw,
            'time': pub_time,  # 新闻发布时间
            'crawl_time': datetime.now(),
            'status': '0'  # 默认状态 0
        }
    except:
        return None


def _run_sync(keywords, days, interval, task_id):
    end_time = datetime.now().timestamp() + days * 86400
    mongo_handler = None
    if task_id:
        mongo_handler = MongoTaskHandler(task_id)
        logger.addHandler(mongo_handler)
        logger.info(f"[Xinhua] Task started. TaskID={task_id}")

    opt = webdriver.ChromeOptions()
    opt.add_argument("--headless=new")
    opt.add_argument("--no-sandbox")
    opt.add_argument("--disable-gpu")
    try:
        opt.add_argument(f"--user-agent={UserAgent().random}")
    except:
        pass
    if settings.HTTP_PROXY: opt.add_argument(f"--proxy-server={settings.HTTP_PROXY}")

    driver = None
    try:
        driver = webdriver.Chrome(service=Service(settings.DRIVER_PATH), options=opt)

        while datetime.now().timestamp() < end_time:
            if task_id and TaskManager.should_stop(task_id):
                logger.info("[Xinhua] Stop signal received.")
                break

            for kw in keywords:
                if task_id and TaskManager.should_stop(task_id): break
                _crawl_one(driver, kw, task_id)

            if datetime.now().timestamp() >= end_time: break
            if task_id and TaskManager.should_stop(task_id): break
            time.sleep(interval * 60)

    except Exception as e:
        logger.error(f"[Xinhua] Error: {e}")
    finally:
        if driver: driver.quit()
        if mongo_handler:
            mongo_handler.flush()
            logger.removeHandler(mongo_handler)


def _crawl_one(driver, kw, task_id):
    logger.info(f"[Xinhua] Searching: {kw}")
    try:
        driver.get("https://so.news.cn/")
        inp = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, "input")))
        inp.clear()
        inp.send_keys(kw)
        driver.find_element(By.CLASS_NAME, "search-button").click()

        valid = 0
        while valid < settings.CRAWL_LIMIT:
            if task_id and TaskManager.should_stop(task_id): break

            try:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "item")))
                items = driver.find_elements(By.CLASS_NAME, "item")
            except:
                break

            for item in items:
                if valid >= settings.CRAWL_LIMIT: break
                if task_id and TaskManager.should_stop(task_id): break
                try:
                    link = item.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    if col.find_one({'url': link}): continue

                    driver.execute_script("window.open(arguments[0])", link)
                    driver.switch_to.window(driver.window_handles[-1])
                    time.sleep(2)

                    detail = extract_detail(driver, kw)
                    if detail:
                        img_ids = []
                        for iurl in detail['images']:
                            fid = download_to_gridfs(iurl)
                            if fid: img_ids.append(fid)
                        detail['images'] = img_ids
                        col.insert_one(detail)
                        valid += 1
                        logger.info(f"   [{valid}] Saved: {detail['title'][:10]}...")

                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                except:
                    if len(driver.window_handles) > 1:
                        driver.close()
                        driver.switch_to.window(driver.window_handles[0])

            try:
                driver.find_element(By.CLASS_NAME, "ant-pagination-next").click()
            except:
                break
    except:
        pass


async def run_task(keywords, days, interval, task_id=None):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_sync, keywords, days, interval, task_id)