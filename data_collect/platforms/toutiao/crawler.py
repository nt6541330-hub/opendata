import random
import time
import io
import os
import traceback
from datetime import datetime, timedelta
from urllib.parse import quote

import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 引入项目统一配置和工具
from config.settings import settings
from common.db import get_toutiao_storage
from common.utils import get_logger
from common.task_manager import TaskManager, MongoTaskHandler

logger = get_logger(__name__)
coll, grid_fs = get_toutiao_storage()

# 浏览器 UA 配置
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/138.0.7204.157 Safari/537.36")


# ---------------- 工具函数 ----------------

def human_scroll(driver, total=3000, step=300):
    """模拟真人滚动"""
    for y in range(0, total, step):
        driver.execute_script(f"window.scrollTo(0, {y})")
        time.sleep(random.uniform(0.3, 0.7))


def download_to_gridfs(url):
    """下载图片并存入 GridFS"""
    try:
        # 使用配置中的代理
        proxies = {"http": settings.HTTP_PROXY, "https": settings.HTTP_PROXY} if settings.HTTP_PROXY else None

        resp = requests.get(url, timeout=10, headers={"User-Agent": UA}, proxies=proxies)
        resp.raise_for_status()
        return grid_fs.put(io.BytesIO(resp.content), filename=url.split("/")[-1][:50])
    except Exception as e:
        logger.warning(f"图片下载失败: {url} {e}")
        return None


def url_exists(url):
    return coll.count_documents({"url": url}, limit=1) > 0


def grab_and_save(driver, keyword):
    """抓取详情页并入库"""
    try:
        WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.CLASS_NAME, "article-content")))
        url = driver.current_url
        if url_exists(url):
            logger.info(f"已存在，跳过：{url}")
            return False

        try:
            title = driver.find_element(By.TAG_NAME, "h1").text.strip()
        except:
            title = "No Title"

        # 时间解析
        dt_obj = None
        try:
            meta = driver.find_element(By.CSS_SELECTOR, "div.article-meta span").text.strip()
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    dt_obj = datetime.strptime(meta, fmt)
                    break
                except ValueError:
                    continue
        except:
            pass

        if dt_obj is None:
            dt_obj = datetime.now()

        # 正文提取
        try:
            paragraphs = driver.find_elements(By.CSS_SELECTOR, "div.article-content p")
            content = "\n".join(p.text.strip() for p in paragraphs)
        except:
            content = ""

        # 图片和视频
        img_elements = driver.find_elements(By.CSS_SELECTOR, "div.article-content img")
        video_elements = driver.find_elements(By.CSS_SELECTOR, "div.article-content video")

        images = []
        for img in img_elements:
            src = img.get_attribute("src")
            if src and not src.startswith("data:image"):
                fid = download_to_gridfs(src)
                if fid: images.append(fid)

        videos = []
        for video in video_elements:
            src = video.get_attribute("src") or video.get_attribute("data-src")
            if src:
                fid = download_to_gridfs(src)
                if fid: videos.append(fid)

        doc = {
            "url": url,
            "source": "今日头条",
            "title": title,
            "time": dt_obj,
            "content": content,
            "images": images,
            "videos": videos,
            "crawl_time": datetime.now(),
            "keyword": keyword,
            "status": "0"
        }
        coll.insert_one(doc)
        logger.info(f"已入库：{title[:30]}")
        return True
    except Exception as e:
        logger.error(f"抓取详情页异常: {e}")
        return False


# ========== 抓取逻辑核心 ==========

def _crawl_one_keyword(driver, keyword: str):
    """针对单个关键词的抓取流程"""
    logger.info(f"[Toutiao] >>> 开始搜索关键词: {keyword}")
    driver.get(f"https://so.toutiao.com/search?dvpf=pc&source=input&keyword={quote(keyword)}")
    time.sleep(5)

    try:
        WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, '//a[text()="资讯"]'))).click()
        time.sleep(3)
    except Exception as e:
        logger.warning(f"点击资讯标签失败: {e}")
        return

    page_count = 0
    total_saved = 0

    # 限制每个关键词最多抓 20 条
    while total_saved < 20:
        page_count += 1
        logger.info(f"[Toutiao] {keyword} 第 {page_count} 页 (已存 {total_saved}/20)")

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.result-content[data-i]')))
            cards = driver.find_elements(By.CSS_SELECTOR, 'div.result-content[data-i]')
        except:
            logger.info("未找到更多结果卡片，停止当前关键词。")
            break

        for idx, card in enumerate(cards):
            if total_saved >= 20:
                logger.info("已达 20 条上限，停止。")
                return

            retry = 0
            success = False
            while retry < 3 and not success:
                try:
                    # 尝试定位链接
                    try:
                        a = card.find_element(By.CSS_SELECTOR, 'a[href*="jump?url="]')
                    except:
                        a = card.find_element(By.TAG_NAME, "a")

                    driver.execute_script("arguments[0].click();", a)
                    time.sleep(random.uniform(1.2, 2.8))

                    # 切换到新窗口
                    driver.switch_to.window(driver.window_handles[-1])
                    success = grab_and_save(driver, keyword)
                    if success:
                        total_saved += 1

                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    time.sleep(random.uniform(1.2, 2.8))
                    break
                except Exception as e:
                    retry += 1
                    logger.warning(f"点击或抓取失败 (重试 {retry}/3): {e}")
                    # 确保异常时窗口切回主页
                    if len(driver.window_handles) > 1:
                        driver.switch_to.window(driver.window_handles[-1])
                        driver.close()
                        driver.switch_to.window(driver.window_handles[0])

                    if retry >= 3:
                        logger.warning("重试次数过多，跳过此条。")
                        break

            if total_saved >= 20: return

        human_scroll(driver, total=3500)
        time.sleep(random.uniform(1.2, 2.8))

        # 翻页
        try:
            next_btn = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//a[.//span[text()="下一页"]]'))
            )
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(3)
        except:
            logger.info("已到末页，结束当前关键字。")
            break


# ========== 唯一入口 (带任务管理) ==========

def run(keywords: list[str], days: float, interval_min: int, task_id: str) -> None:
    """
    执行采集任务
    :param task_id: 任务ID，用于日志入库
    """
    deadline = datetime.now() + timedelta(days=days)

    # 1. 【关键】挂载 MongoDB 日志处理器
    # 这样所有 logger.info/warning 都会自动存入数据库
    mongo_handler = MongoTaskHandler(task_id)
    logger.addHandler(mongo_handler)

    logger.info(f"[Toutiao] Task started. TaskID={task_id}, Keywords={keywords}")

    # 2. 浏览器配置
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")  # 服务器必须无头
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(f"--user-agent={UA}")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    if settings.HTTP_PROXY:
        options.add_argument(f"--proxy-server={settings.HTTP_PROXY}")

    try:
        driver = webdriver.Chrome(service=Service(settings.DRIVER_PATH), options=options)

        # 注入 stealth.js (防爬虫检测)
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })
        if os.path.exists(settings.STEALTH_JS_PATH):
            with open(settings.STEALTH_JS_PATH, "r", encoding="utf-8") as f:
                stealth_local = f.read()
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": stealth_local})

        logger.info("[Toutiao] Browser launched successfully.")

        # 3. 循环采集
        round_count = 0
        while datetime.now() < deadline:
            round_count += 1
            logger.info(f"[Toutiao] Round {round_count} Start")

            for kw in keywords:
                _crawl_one_keyword(driver, kw)

            # 轮次间隔
            base_seconds = interval_min * 60
            jitter = random.uniform(-base_seconds * 0.2, base_seconds * 0.2)
            sleep_seconds = max(30, base_seconds + jitter)
            logger.info(f"[Toutiao] Round {round_count} finished. Sleep {sleep_seconds:.1f}s...")
            time.sleep(sleep_seconds)

        # 4. 任务完成
        TaskManager.finish_task(task_id, status="finished")
        logger.info("[Toutiao] Task finished successfully.")

    except Exception as e:
        # 5. 任务失败
        error_msg = f"{e}\n{traceback.format_exc()}"
        logger.error(f"[Toutiao] Task Critical Error: {e}")
        TaskManager.finish_task(task_id, status="failed", error=error_msg)

    finally:
        try:
            driver.quit()
        except:
            pass
        # 移除日志处理器，避免污染其他任务
        mongo_handler.flush()
        logger.removeHandler(mongo_handler)