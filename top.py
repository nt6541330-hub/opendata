import random, time, requests, io, sys
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import quote
import pymongo
from gridfs import GridFS


# 加载本地 stealth.min.js 文件
STEALTH_PATH = r"/hesiqi/stealth.min.js"
with open(STEALTH_PATH, "r", encoding="utf-8") as f:
    stealth_local = f.read()

# 数据库配置
DB_URI = "mongodb://admin:123456@127.0.0.1:8800/"
DB_NAME = "test"
COLL_NAME = "toutiao"
BUCKET_NAME = "fs_toutiao"

client = pymongo.MongoClient(DB_URI)
db = client[DB_NAME]
coll = db[COLL_NAME]
grid_fs = GridFS(db, collection=BUCKET_NAME)

# 浏览器初始化
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/138.0.7204.157 Safari/537.36")  # 修改了 Chrome 版本
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument(f"--user-agent={UA}")
options.add_argument("--accept-language=zh-CN,zh;q=0.9,en;q=0.8")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
# 浏览器驱动所在位置
DRIVER_PATH = r"/usr/local/bin/chromedriver"  # 修改为适合 Windows 的路径
service = Service(DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)
driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
})
driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": stealth_local})

# 首屏先刷 PC 首页
driver.get("https://so.toutiao.com/search?dvpf=pc ")
time.sleep(3)

# 工具函数
def human_scroll(driver, total=3000, step=300):
    for y in range(0, total, step):
        driver.execute_script(f"window.scrollTo(0, {y})")
        time.sleep(random.uniform(0.3, 0.7))

# 下载图片
def download_to_gridfs(url):
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": UA})
        resp.raise_for_status()
        return grid_fs.put(io.BytesIO(resp.content), filename=url.split("/")[-1][:50])
    except Exception as e:
        print("图片下载失败:", e)
        return None

def url_exists(url):
    return coll.count_documents({"url": url}, limit=1) > 0

def grab_and_save(driver,keyword):
    WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.CLASS_NAME, "article-content")))
    url = driver.current_url
    if url_exists(url):
        print("   已存在，跳过：", url)
        return False

    title = driver.find_element(By.TAG_NAME, "h1").text.strip()
    meta = driver.find_element(By.CSS_SELECTOR, "div.article-meta span").text.strip()
    dt_obj = None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt_obj = datetime.strptime(meta, fmt)
            break
        except ValueError:
            continue
    if dt_obj is None:
        dt_obj = datetime.now()

    paragraphs = driver.find_elements(By.CSS_SELECTOR, "div.article-content p")
    content = "\n".join(p.text.strip() for p in paragraphs)

    img_elements = driver.find_elements(By.CSS_SELECTOR, "div.article-content img")
    video_elements = driver.find_elements(By.CSS_SELECTOR, "div.article-content video")

    images = []
    for img in img_elements:
        src = img.get_attribute("src")
        if src and not src.startswith("data:image"):
            fid = download_to_gridfs(src)
            if fid:
                images.append(fid)

    videos = []
    for video in video_elements:
        src = video.get_attribute("src") or video.get_attribute("data-src")
        if src:
            fid = download_to_gridfs(src)  # ✅ 你原来这个函数就能直接复用
            if fid:
                videos.append(fid)

    doc = {
        "url": url,
        "title": title,
        "time": dt_obj,
        "content": content,
        "images": images,
        "videos": videos,
        "crawl_time": datetime.now(),
        "source": "今日头条",
        "keyword": keyword
    }
    coll.insert_one(doc)
    print("   ↑ 已入库：", title[:30])
    return True

# ------------------- 输入参数 -------------------
keywords_input = input("请输入关键字（空格分隔）：").strip()
if not keywords_input:
    sys.exit("关键字不能为空")
keywords = keywords_input.split()

try:
    days = float(input("持续抓取天数（可小数，如 0.5=12h）："))
    interval_minutes = int(input("抓取间隔分钟："))
except ValueError:
    sys.exit("请输入合法数字")

deadline = datetime.now() + timedelta(days=days)
print(f"【参数确认】关键字：{keywords} | 持续：{days}天 | 间隔：{interval_minutes}分钟")
print("终止时间：", deadline.strftime("%Y-%m-%d %H:%M:%S"))

# ------------------- 主循环 -------------------
round_count = 0
while datetime.now() < deadline:
    round_count += 1
    print(f"\n>>>>>>>> 第 {round_count} 轮抓取开始 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for kw in keywords:
        print(f"\n>>> 关键字 【{kw}】 开始（限25条）")
        driver.get(f"https://so.toutiao.com/search?dvpf=pc&source=input&keyword= {quote(kw)}")
        time.sleep(5)
        WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, '//a[text()="资讯"]'))).click()
        time.sleep(3)

        page_count = 0
        no_next_page_count = 0
        total_saved = 0

        while total_saved < 20:
            page_count += 1
            print(f"===== 第 {page_count} 页 =====")
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.result-content[data-i]')))
            cards = driver.find_elements(By.CSS_SELECTOR, 'div.result-content[data-i]')

            for idx, card in enumerate(cards, 1):
                if total_saved >= 25:
                    print("已达25条，提前结束当前关键字")
                    break
                retry = 0
                success = False
                while retry < 3 and not success:
                    try:
                        try:
                            a = card.find_element(By.CSS_SELECTOR, 'a[href*="jump?url="]')
                        except:
                            a = card.find_element(By.TAG_NAME, "a")
                        title = a.text.strip()
                        # 取出即将点击的 href
                        href = a.get_attribute('href')
                        print(f"{idx:02d} 即将点击 URL：{href}")
                        print(f"{idx:02d} 点击：{title[:30]}...")
                        driver.execute_script("arguments[0].click();", a)
                        # time.sleep(2)
                        time.sleep(random.uniform(1.2, 2.8))

                        driver.switch_to.window(driver.window_handles[-1])
                        grab_and_save(driver, kw)
                        total_saved += 1
                        driver.close()
                        driver.switch_to.window(driver.window_handles[0])
                        # time.sleep(1)
                        time.sleep(random.uniform(1.2, 2.8))
                        success = True
                    except Exception as e:
                        retry += 1
                        print(f"  点击失败，第{retry}次重试：{e}")
                        if retry >= 3:
                            print(f"  跳过该条新闻")
                        continue
                if total_saved >= 25:
                    break

            if total_saved >= 25:
                break

            human_scroll(driver, total=3500)
            # time.sleep(2)
            time.sleep(random.uniform(1.2, 2.8))
            try:
                next_btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//a[.//span[text()="下一页"]]'))
                )
                driver.execute_script("arguments[0].click();", next_btn)
                time.sleep(3)
                no_next_page_count = 0
            except:
                no_next_page_count += 1
                print(f"未找到“下一页”按钮，第 {no_next_page_count} 次")
                if no_next_page_count >= 3:
                    print("连续3次未找到“下一页”，结束当前关键字")
                    break
                else:
                    print("等待后再次尝试...")
                    time.sleep(3)
                    continue

    # 本轮关键字全部跑完，再判断下次间隔/截止
    if datetime.now() >= deadline:
        print("已到达截止时间，本轮结束后退出")
        break
    # sleep_seconds = interval_minutes * 60
    # print(f"本轮完成，{interval_minutes}分钟后下一轮（按分钟精确等待）...")
    # time.sleep(sleep_seconds)
    base_seconds = interval_minutes * 60
    jitter = random.uniform(-base_seconds * 0.2, base_seconds * 0.2)  # ±20 % 浮动
    sleep_seconds = max(30, base_seconds + jitter)  # 最少 30 s，防止负值
    print(f"本轮完成，{sleep_seconds / 60:.1f} 分钟后下一轮...")
    time.sleep(sleep_seconds)


print("\n全部持续抓取任务完成！")
driver.quit()

