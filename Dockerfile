# 1. 使用轻量级 Python 3.10 镜像
FROM python:3.10-slim

# 2. 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 3. 安装系统依赖和 Chrome (用于 Selenium)
RUN apt-get update && apt-get install -y \
    wget gnupg unzip libgl1 libglib2.0-0 curl \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. 安装 Chromedriver (自动匹配版本)
RUN CHROME_VERSION=$(google-chrome --version | awk '{print $3}' | awk -F'.' '{print $1}') && \
    # 这里使用通用获取 Latest Stable 的方法，如果版本对不上可以手动指定版本号
    DRIVER_VERSION=$(curl -s "https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE") && \
    wget -q "https://storage.googleapis.com/chrome-for-testing-public/${DRIVER_VERSION}/linux64/chromedriver-linux64.zip" -O /tmp/chromedriver.zip && \
    unzip /tmp/chromedriver.zip -d /tmp/ && \
    mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver && \
    chmod +x /usr/local/bin/chromedriver && \
    rm -rf /tmp/chromedriver*

# 5. 安装 Python 依赖
COPY requirements.txt .
# 强制安装 CPU 版 torch 以减小体积 (如果代码必须依赖 torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu || true
RUN pip install --no-cache-dir -r requirements.txt

# 6. 复制项目代码
COPY . .

# 7. 暴露端口
EXPOSE 8801

# 8. 启动命令
CMD ["python", "main.py"]