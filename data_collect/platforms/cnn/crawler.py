import time
import traceback  # 【新增】用于打印完整报错
from datetime import datetime
from typing import List, Optional
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 注意：这里保留了 langchain 的引用
try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    pass

from config.settings import settings
from common.db import get_cnn_storage
from common.utils import get_logger
# 【新增】任务管理模块
from common.task_manager import TaskManager, MongoTaskHandler

logger = get_logger(__name__)
collection, fs = get_cnn_storage()


def translate_text(text: str, target_lang="zh") -> str:
    """简单的 LLM 翻译占位"""
    try:
        # 如果没有安装 langchain 或没有 Ollama 服务，这里会报错或卡住
        # 建议生产环境加个开关控制是否开启翻译
        llm = ChatOllama(model="qwen3:14b", temperature=0.1)
        prompt = PromptTemplate.from_template("Translate to {lang}: {text}")
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"lang": "Chinese" if target_lang == "zh" else "English", "text": text})
    except:
        return text


# 【修改】run 函数接收 task_id
def run(keywords: list[str], days: float, interval_min: float, task_id: str) -> None:
    deadline = time.time() + days * 86400

    # 1. 【核心】挂载 MongoDB 日志处理器
    mongo_handler = MongoTaskHandler(task_id)
    logger.addHandler(mongo_handler)

    logger.info(f"[CNN] Task started. TaskID={task_id}, Keywords={keywords}")

    opt = webdriver.ChromeOptions()
    opt.add_argument("--headless=new")
    # 【新增】Linux 服务器必备参数，防止崩溃
    opt.add_argument("--no-sandbox")
    opt.add_argument("--disable-gpu")
    opt.add_argument("--disable-dev-shm-usage")

    if settings.HTTP_PROXY:
        opt.add_argument(f"--proxy-server={settings.HTTP_PROXY}")

    driver = None
    try:
        service = Service(settings.DRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=opt)
        logger.info("[CNN] Browser launched successfully.")

        while time.time() < deadline:
            for kw in keywords:
                try:
                    kw_en = translate_text(kw, "en")  # 中译英
                    logger.info(f"[CNN] Searching: {kw_en} (Original: {kw})")

                    driver.get(f"https://edition.cnn.com/search?q={kw_en}&size=10&types=article")
                    time.sleep(5)  # 等待加载

                    # 抓取逻辑
                    links = []
                    try:
                        # 尝试定位卡片
                        elems = driver.find_elements(By.CSS_SELECTOR, "div.card")
                        for el in elems:
                            try:
                                link = el.find_element(By.TAG_NAME, "a").get_attribute("href")
                                if link: links.append(link)
                            except:
                                pass
                    except Exception as e:
                        logger.warning(f"[CNN] Failed to find cards: {e}")

                    # 限制抓取数量，避免单次任务过久
                    for url in links[:settings.CRAWL_LIMIT]:
                        if collection.find_one({"url": url}):
                            continue

                        try:
                            driver.get(url)
                            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))

                            title = driver.find_element(By.TAG_NAME, "h1").text
                            # 尝试获取正文，CNN 结构可能变化，增加容错
                            content = ""
                            try:
                                content = driver.find_element(By.TAG_NAME, "article").text
                            except:
                                # 备用选择器
                                content_bs = driver.find_elements(By.CSS_SELECTOR, ".article__content")
                                if content_bs: content = content_bs[0].text

                            if not content:
                                logger.warning(f"[CNN] No content found for {url}")
                                continue

                            doc = {
                                "title": title,
                                "title_zh": translate_text(title),
                                "content": content,
                                "content_zh": translate_text(content[:500]),  # 只翻译前500字演示，节省资源
                                "url": url,
                                "source": "CNN",
                                "keyword": kw,
                                "crawl_time": datetime.now()
                            }
                            collection.insert_one(doc)
                            logger.info(f"[CNN] Saved: {title[:20]}...")

                        except Exception as e:
                            logger.warning(f"[CNN] Detail error: {e}")

                except Exception as e:
                    logger.error(f"[CNN] Keyword Error {kw}: {e}")

            # 轮次间隔
            sleep_time = interval_min * 60
            logger.info(f"[CNN] Round finished. Sleeping {sleep_time}s...")
            time.sleep(sleep_time)

        # 任务正常结束
        TaskManager.finish_task(task_id, status="finished")
        logger.info("[CNN] Task finished successfully.")

    except Exception as e:
        # 任务异常
        error_msg = f"{e}\n{traceback.format_exc()}"
        logger.error(f"[CNN] Task Critical Error: {e}")
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