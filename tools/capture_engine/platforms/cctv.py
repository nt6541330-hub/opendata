import time
import requests
import traceback
import re
from datetime import datetime
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import asyncio

from config.settings import settings
from common.db import get_cctv_storage
from common.utils import get_logger
from common.task_manager import TaskManager, MongoTaskHandler

logger = get_logger(__name__)
coll, grid_fs = get_cctv_storage()


def save_to_gridfs(url_list):
    ids = []
    for u in url_list:
        try:
            r = requests.get(u, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                fid = grid_fs.put(BytesIO(r.content), filename=u.split('/')[-1], url=u)
                ids.append(fid)
        except:
            pass
    return ids


def parse_cctv_time(info_text):
    try:
        m = re.search(r'(\d{4}[年-]\d{1,2}[月-]\d{1,2})', info_text)
        if m:
            tm_str = m.group(1).replace('年', '-').replace('月', '-').replace('日', '')
            return datetime.strptime(tm_str, '%Y-%m-%d')
    except:
        pass
    return datetime.now()


def _run_sync(keywords, days, interval, task_id):
    end_time = datetime.now().timestamp() + days * 24 * 3600
    mongo_handler = None

    if task_id:
        mongo_handler = MongoTaskHandler(task_id)
        logger.addHandler(mongo_handler)
        logger.info(f"[CCTV] Task started. TaskID={task_id}")

    opt = webdriver.ChromeOptions()
    opt.add_argument('--headless')
    opt.add_argument('--no-sandbox')
    opt.add_argument('--disable-gpu')
    if settings.HTTP_PROXY: opt.add_argument(f'--proxy-server={settings.HTTP_PROXY}')

    driver = None
    try:
        driver = webdriver.Chrome(service=Service(settings.DRIVER_PATH), options=opt)

        while datetime.now().timestamp() < end_time:
            if task_id and TaskManager.should_stop(task_id):
                logger.info("[CCTV] Stop signal received.")
                break

            for kw in keywords:
                if task_id and TaskManager.should_stop(task_id): break
                _crawl_one(driver, kw, task_id)

            if datetime.now().timestamp() >= end_time: break
            if task_id and TaskManager.should_stop(task_id): break
            time.sleep(interval * 60)

    except Exception as e:
        logger.error(f"[CCTV] Error: {e}")
    finally:
        if driver: driver.quit()
        if mongo_handler:
            mongo_handler.flush()
            logger.removeHandler(mongo_handler)


def _crawl_one(driver, kw, task_id):
    try:
        logger.info(f"[CCTV] Searching: {kw}")
        driver.get('https://search.cctv.com/index.php')
        time.sleep(2)
        try:
            driver.find_element(By.ID, 'web').click()
        except:
            pass

        driver.find_element(By.ID, 'search_qtext').clear()
        driver.find_element(By.ID, 'search_qtext').send_keys(kw)
        driver.find_element(By.XPATH, '//a[@onclick="searchForm_submit();"]').click()
        time.sleep(3)

        count = 0
        while count < settings.CRAWL_LIMIT:
            if task_id and TaskManager.should_stop(task_id): break

            try:
                links = driver.find_elements(By.CSS_SELECTOR, 'div.outer ul span[lanmu1]')
                urls = [l.get_attribute('lanmu1') for l in links]
            except:
                break
            if not urls: break

            for url in urls:
                if count >= settings.CRAWL_LIMIT: break
                if task_id and TaskManager.should_stop(task_id): break

                if coll.find_one({'url': url}): continue

                driver.execute_script(f'window.open("{url}");')
                driver.switch_to.window(driver.window_handles[-1])
                try:
                    title = driver.find_element(By.TAG_NAME, 'h1').text
                    content = driver.find_element(By.ID, 'content_area').text
                    imgs = [i.get_attribute('src') for i in driver.find_elements(By.TAG_NAME, 'img')]

                    info_text = ""
                    try:
                        info_text = driver.find_element(By.CLASS_NAME, 'info').text
                    except:
                        pass
                    pub_time = parse_cctv_time(info_text)

                    img_ids = save_to_gridfs(imgs)

                    coll.insert_one({
                        "title": title,
                        "url": url,
                        "content": content,
                        "images": img_ids,
                        "source": "CCTV",
                        "keyword": kw,
                        "time": pub_time,  # 新闻发布时间
                        "crawl_time": datetime.now(),
                        "status": "0"  # 默认状态 0
                    })
                    count += 1
                    logger.info(f"   [{count}] Saved: {title[:10]}...")
                except:
                    pass
                finally:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])

            try:
                driver.find_element(By.CLASS_NAME, 'page-next').click()
                time.sleep(2)
            except:
                break
    except:
        pass


async def run_task(keywords, days, interval, task_id=None):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_sync, keywords, days, interval, task_id)