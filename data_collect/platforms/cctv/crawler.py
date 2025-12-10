import time
import re
import requests
import traceback  # 【新增】用于打印报错堆栈
from datetime import datetime
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException

from config.settings import settings
from common.db import get_cctv_storage
from common.utils import get_logger
# 【新增】任务管理模块
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
        except Exception as e:
            logger.warning(f'下载失败，跳过：{u} {e}')
    return ids


# 【修改】run 函数接收 task_id
def run(keywords: list[str], days: float, interval_min: int, task_id: str) -> None:
    total_seconds = days * 24 * 3600
    interval_seconds = interval_min * 60
    end_time = datetime.now().timestamp() + total_seconds
    round_no = 0

    # 1. 【核心】挂载 MongoDB 日志处理器
    mongo_handler = MongoTaskHandler(task_id)
    logger.addHandler(mongo_handler)

    logger.info(f"[CCTV] Task started. TaskID={task_id}, Keywords={keywords}")

    try:
        while datetime.now().timestamp() < end_time:
            round_no += 1
            logger.info(f'[CCTV] Round {round_no} start')
            for kw in keywords:
                _crawl_one_keyword(kw)

            left = end_time - datetime.now().timestamp()
            if left <= 0: break
            logger.info(f"[CCTV] Sleep {min(interval_seconds, left)}s")
            time.sleep(min(interval_seconds, left))

        # 任务正常结束
        TaskManager.finish_task(task_id, status="finished")
        logger.info("[CCTV] Task finished successfully.")

    except Exception as e:
        # 任务异常
        error_msg = f"{e}\n{traceback.format_exc()}"
        logger.error(f"[CCTV] Task Critical Error: {e}")
        TaskManager.finish_task(task_id, status="failed", error=error_msg)

    finally:
        # 移除日志处理器
        mongo_handler.flush()
        logger.removeHandler(mongo_handler)


def _crawl_one_keyword(kw: str):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')  # CCTV 反爬较弱，通常旧版 headless 即可，也可改为 --headless=new
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-dev-shm-usage')
    if settings.HTTP_PROXY:
        chrome_options.add_argument(f'--proxy-server={settings.HTTP_PROXY}')

    driver = None
    try:
        driver = webdriver.Chrome(service=Service(settings.DRIVER_PATH), options=chrome_options)
        driver.get('https://search.cctv.com/index.php')
        time.sleep(2)

        # 搜索交互
        try:
            driver.find_element(By.ID, 'web').click()  # 点击网页搜索
        except:
            pass

        input_box = driver.find_element(By.ID, 'search_qtext')
        input_box.clear()
        input_box.send_keys(kw)
        time.sleep(1)
        driver.find_element(By.XPATH, '//a[@onclick="searchForm_submit();"]').click()
        time.sleep(3)

        total_count = 0
        page_no = 1

        while total_count < settings.CRAWL_LIMIT:
            logger.info(f'[CCTV] {kw} page {page_no}')
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            try:
                # 定位搜索结果列表
                ul = driver.find_element(By.CSS_SELECTOR, 'div.outer ul')
                spans = ul.find_elements(By.XPATH, './/span[@lanmu1]')
                links = [s.get_attribute('lanmu1').strip() for s in spans]
            except Exception:
                break  # 没找到结果

            main_window = driver.current_window_handle
            for url in links:
                if total_count >= settings.CRAWL_LIMIT: return

                if coll.find_one({'url': url}):
                    continue

                total_count += 1
                driver.execute_script(f'window.open("{url}");')
                time.sleep(1)
                driver.switch_to.window(driver.window_handles[-1])

                try:
                    # 尝试定位正文容器
                    root = None
                    for cls in ['content_18313', 'content_19568', 'cnt_bd']:
                        try:
                            root = driver.find_element(By.CLASS_NAME, cls)
                            break
                        except:
                            continue

                    if not root:
                        driver.close()
                        driver.switch_to.window(main_window)
                        continue

                    title = driver.find_element(By.TAG_NAME, 'h1').text.strip()

                    # 解析时间
                    pub_time = datetime.now()
                    try:
                        info_text = driver.find_element(By.CLASS_NAME, 'info').text
                        # 简单正则匹配日期
                        m = re.search(r'(\d{4}[年-]\d{1,2}[月-]\d{1,2})', info_text)
                        if m:
                            tm_str = m.group(1).replace('年', '-').replace('月', '-')
                            pub_time = datetime.strptime(tm_str, '%Y-%m-%d')
                    except:
                        pass

                    content = root.text.strip()

                    # 提取多媒体
                    images = []
                    for img in root.find_elements(By.TAG_NAME, 'img'):
                        src = img.get_attribute('src')
                        if src and not src.startswith('data:'):
                            images.append(src)

                    image_ids = save_to_gridfs(images)

                    doc = {
                        'title': title,
                        'source': '央视网',
                        'content': content,
                        'time': pub_time,
                        'images': image_ids,
                        'crawl_time': datetime.now(),
                        'keyword': kw,
                        'url': url,
                        'status': '0'
                    }
                    coll.insert_one(doc)
                    logger.info(f'[CCTV] Saved: {title[:20]}')

                except Exception as e:
                    logger.warning(f"Page error: {e}")

                driver.close()
                driver.switch_to.window(main_window)
                time.sleep(0.5)

            # 翻页
            try:
                next_btn = driver.find_element(By.CLASS_NAME, 'page-next')
                if "下一页" not in next_btn.text: break
                next_btn.click()
                page_no += 1
                time.sleep(3)
            except:
                break

    except Exception as e:
        logger.error(f"[CCTV] Keyword Error {kw}: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass