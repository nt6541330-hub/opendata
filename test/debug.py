import requests
import json
import os

# 1. 配置你的 Cookie 文件路径
COOKIE_FILE = "open_source_data/weibo_cookies.json"
# 2. 测试关键词
KEYWORD = "大模型"


def test_weibo():
    if not os.path.exists(COOKIE_FILE):
        print(f"❌ 错误：找不到 Cookie 文件: {COOKIE_FILE}")
        return

    # 加载 Cookie
    try:
        with open(COOKIE_FILE, "r", encoding="utf-8") as f:
            cookies_list = json.load(f)
        # 转换为 requests 格式的 dict
        cookie_dict = {c['name']: c['value'] for c in cookies_list if 'name' in c and 'value' in c}
        print(f"✅ 已加载 {len(cookie_dict)} 个 Cookie 字段")
    except Exception as e:
        print(f"❌ Cookie 文件解析失败: {e}")
        return

    # 构造请求
    url = "https://m.weibo.cn/api/container/getIndex"
    params = {
        "containerid": f"100103type=1&q={KEYWORD}",
        "page_type": "searchall",
        "page": 1
    }

    # 模拟真实手机 UA
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.91 Mobile Safari/537.36",
        "Referer": f"https://m.weibo.cn/search?containerid=100103type%3D1%26q%3D{KEYWORD}",
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest"
    }

    print(f"\n>>> 正在请求微博搜索接口: {KEYWORD} ...")
    try:
        # 如果你有代理，可以在这里加 proxies={"http": "...", "https": "..."}
        resp = requests.get(url, params=params, cookies=cookie_dict, headers=headers, timeout=10)

        print(f"状态码: {resp.status_code}")

        # 检查是否被重定向到登录页
        if "passport.weibo.cn/signin" in resp.url:
            print("❌ 失败：请求被重定向到了登录页，Cookie 已失效！")
            return

        try:
            data = resp.json()
            if data.get("ok") == 1:
                cards = data.get("data", {}).get("cards", [])
                print(f"✅ 成功！获取到 {len(cards)} 条微博数据")
                if cards:
                    first_text = cards[0].get("mblog", {}).get("text", "无正文")
                    print(f"   示例: {first_text[:50]}...")
            else:
                print(f"⚠️ 接口返回 ok!=1，可能是被风控或无结果。响应: {data}")
        except json.JSONDecodeError:
            print(f"❌ 解析失败，返回的不是 JSON。前 200 字符:\n{resp.text[:200]}")

    except Exception as e:
        print(f"❌ 请求异常: {e}")


if __name__ == "__main__":
    test_weibo()