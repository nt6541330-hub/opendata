import base64
import io
import random
import time
import re
import traceback  # 【新增】用于打印报错堆栈
from datetime import datetime
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from fake_useragent import UserAgent

from config.settings import settings
from common.db import get_xinhua_storage
from common.utils import get_logger
# 【新增】任务管理模块
from common.task_manager import TaskManager, MongoTaskHandler

logger = get_logger(__name__)
col, fs = get_xinhua_storage()


def download_to_gridfs(url, media_type):
    if not url: return None
    try:
        if url.startswith('data:'):
            return None

        proxies = {"http": settings.HTTP_PROXY, "https": settings.HTTP_PROXY} if settings.HTTP_PROXY else None
        r = requests.get(url, timeout=20, stream=True, proxies=proxies)
        if r.status_code == 200:
            # 使用 upload_from_stream 直接存入 GridFS
            return fs.upload_from_stream(url.split('/')[-1], io.BytesIO(r.content))
    except:
        pass
    return None


def extract_detail(driver, kw):
    try:
        title = driver.find_element(By.CSS_SELECTOR, 'h1 span.title').text.strip()
        content = driver.find_element(By.ID, 'detail').text.strip()
    except:
        return None

    imgs = []
    try:
        for img in driver.find_elements(By.CSS_SELECTOR, 'img'):
            src = img.get_attribute('src')
            if src: imgs.append(src)
    except:
        pass

    return {
        'title': title,
        'source': '新华社',
        'content': content,
        'images': imgs,
        'url': driver.current_url,
        'keyword': kw,
        'crawl_time': datetime.now(),
        'status': '0'
    }


# 【修改】run 函数接收 task_id
def run(keywords: list[str], days: float, interval_min: float, task_id: str) -> None:
    deadline = time.time() + days * 86400

    # 1. 【核心】挂载 MongoDB 日志处理器
    mongo_handler = MongoTaskHandler(task_id)
    logger.addHandler(mongo_handler)

    logger.info(f"[Xinhua] Task started. TaskID={task_id}, Keywords={keywords}")

    opt = webdriver.ChromeOptions()
    opt.add_argument("--headless=new")
    opt.add_argument("--no-sandbox")
    opt.add_argument("--disable-gpu")  # Linux 必备
    opt.add_argument("--disable-dev-shm-usage")  # Linux 必备

    ua = UserAgent().random
    opt.add_argument(f"--user-agent={ua}")

    if settings.HTTP_PROXY:
        opt.add_argument(f"--proxy-server={settings.HTTP_PROXY}")

    driver = None
    try:
        service = Service(settings.DRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=opt)
        logger.info("[Xinhua] Browser launched successfully.")

        while time.time() < deadline:
            for kw in keywords:
                logger.info(f"[Xinhua] >>> Searching: {kw}")
                try:
                    driver.get("https://so.news.cn/")

                    try:
                        inp = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, "input")))
                        inp.clear()
                        inp.send_keys(kw)
                        driver.find_element(By.CLASS_NAME, "search-button").click()
                    except Exception as e:
                        logger.warning(f"[Xinhua] Search input failed: {e}")
                        continue

                    # 结果页
                    valid_count = 0
                    main_window = driver.current_window_handle

                    # 限制抓取数量
                    while valid_count < settings.CRAWL_LIMIT:
                        try:
                            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "item")))
                            items = driver.find_elements(By.CLASS_NAME, "item")
                        except:
                            logger.info("[Xinhua] No results found on this page.")
                            break

                        for item in items:
                            if valid_count >= settings.CRAWL_LIMIT: break
                            try:
                                # 定位链接
                                a_tag = item.find_element(By.TAG_NAME, 'a')
                                link = a_tag.get_attribute('href')
                                title_preview = a_tag.text.strip()

                                # 简单去重
                                if col.find_one({'url': link}):
                                    continue

                                driver.execute_script("window.open(arguments[0])", link)
                                driver.switch_to.window(driver.window_handles[-1])
                                time.sleep(2)

                                detail = extract_detail(driver, kw)
                                if detail:
                                    # 处理图片
                                    img_ids = []
                                    for iurl in detail['images']:
                                        fid = download_to_gridfs(iurl, 'image')
                                        if fid: img_ids.append(fid)
                                    detail['images'] = img_ids

                                    col.insert_one(detail)
                                    valid_count += 1
                                    logger.info(f"[Xinhua] Saved: {detail['title'][:15]}...")

                                driver.close()
                                driver.switch_to.window(main_window)
                            except Exception:
                                # 详情页异常处理
                                if len(driver.window_handles) > 1:
                                    driver.close()
                                driver.switch_to.window(main_window)
                                continue

                        # 翻页
                        if valid_count >= settings.CRAWL_LIMIT: break
                        try:
                            next_btn = driver.find_element(By.CLASS_NAME, "ant-pagination-next")
                            # 检查是否禁用
                            if "disabled" in next_btn.get_attribute(
                                    "class") or "aria-disabled" in next_btn.get_attribute("outerHTML"):
                                logger.info("[Xinhua] Reached last page.")
                                break
                            next_btn.click()
                            time.sleep(3)
                        except:
                            break

                    logger.info(f"[Xinhua] Keyword '{kw}' finished. Total saved: {valid_count}")

                except Exception as e:
                    logger.error(f"[Xinhua] Error processing keyword {kw}: {e}")

            # 轮次间隔
            sleep_time = interval_min * 60
            logger.info(f"[Xinhua] Round finished. Sleeping {sleep_time}s ...")
            time.sleep(sleep_time)

        # 任务正常结束
        TaskManager.finish_task(task_id, status="finished")
        logger.info("[Xinhua] Task finished successfully.")

    except Exception as e:
        # 任务异常
        error_msg = f"{e}\n{traceback.format_exc()}"
        logger.error(f"[Xinhua] Task Critical Error: {e}")
        TaskManager.finish_task(task_id, status="failed", error=error_msg)

    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass
        # 移除日志处理器
        mongo_handler.flush()
        logger.removeHandler(mongo_handler)