# open_source_data/common/utils.py
import logging
import sys
import os
import json
import re
import requests
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from bson import ObjectId
from PIL import Image

# 引入统一配置
from config.settings import settings

# ==========================================
# 1. 日志配置 (Logging)
# ==========================================

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "osd.log")


def setup_logging():
    """配置全局日志：同时输出到 控制台 和 文件"""
    # 获取根日志记录器
    root_logger = logging.getLogger()

    # 避免重复添加 handler
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.INFO)

    # 格式：时间 | 级别 | 模块 | 消息
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 处理器 A: 输出到控制台 (Console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 处理器 B: 输出到文件 (File) - 每天午夜轮转一次，保留最近 7 天
    file_handler = TimedRotatingFileHandler(
        filename=LOG_FILE,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


# 初始化配置 (被 import 时自动执行一次)
setup_logging()


def get_logger(name: str) -> logging.Logger:
    """获取带名称的 Logger"""
    return logging.getLogger(name)


# 模块内部使用的 logger
_logger = get_logger(__name__)


# ==========================================
# 2. 时间与序列化 (Time & Serialization)
# ==========================================

def now_iso() -> str:
    """返回当前时间的 ISO 字符串，UTC"""
    return datetime.utcnow().isoformat()


def default_serializer(o):
    """
    JSON序列化辅助函数：
    把 MongoDB 中的 ObjectId、datetime 等转成字符串，防止 json.dump 报错
    """
    if isinstance(o, ObjectId):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


# ==========================================
# 3. 文件与 JSON 操作 (File & JSON)
# ==========================================

def ensure_dir(path: str) -> None:
    """确保目录存在，不存在就创建"""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def read_json(path: str):
    """从文件读取 JSON，没有文件就返回 None"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _logger.error(f"Read JSON failed: {path} error: {e}")
        return None


def write_json(path: str, data) -> None:
    """把对象写到 JSON 文件里，带缩进"""
    try:
        ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=default_serializer)
    except Exception as e:
        _logger.error(f"Write JSON failed: {path} error: {e}")


# ==========================================
# 4. 地理位置服务 (Geocoding)
# ==========================================

def geocode_address(address: str):
    """
    使用配置中的 API 进行地址正向地理编码，返回 "经度,纬度" 字符串。
    """
    if not address:
        return None

    # 从 settings 获取配置，使用 getattr 防止配置未定义时报错
    api_key = getattr(settings, "GOOGLE_MAP_API_KEY", None)
    url = getattr(settings, "GOOGLE_GEOCODE_URL", None)

    if not url:
        # 如果未配置 URL，则跳过（或者可以设置一个默认的 OpenStreetMap URL）
        return None

    params = {'q': address}
    if api_key:
        params['api_key'] = api_key

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data and isinstance(data, list) and len(data) > 0:
            # 适配返回格式 [{'lon': ..., 'lat': ...}]
            item = data[0]
            lon = item.get('lon')
            lat = item.get('lat')
            if lon and lat:
                return f"{lon},{lat}"

        _logger.warning(f"⚠️ 地址匹配失败或无结果：{address}")
        return None

    except requests.exceptions.RequestException as e:
        _logger.error(f"❌ Geocode 请求异常：{e}")
        return None


# ==========================================
# 5. 图像处理 (Image Utils)
# ==========================================

def sniff_mime(header: bytes):
    """根据文件头字节判断图片 MIME 类型"""
    if header.startswith(b'\xFF\xD8\xFF'):
        return 'image/jpeg'
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
        return 'image/gif'
    if len(header) >= 12 and header[:4] == b'RIFF' and header[8:12] == b'WEBP':
        return 'image/webp'
    if header.startswith(b'BM'):
        return 'image/bmp'
    return None


def has_alpha(img: Image.Image) -> bool:
    """判断图片是否包含透明通道"""
    if img.mode in ('RGBA', 'LA'):
        return True
    if img.mode == 'P':
        return 'transparency' in img.info
    return False


# ==========================================
# 6. 文本处理 (Text Utils)
# ==========================================

_BR_RE = re.compile(r"<br\s*/?>", re.I)
_TAG_RE = re.compile(r"<.*?>", re.S)


def strip_html(text: str) -> str:
    """简单去掉 HTML 标签并处理换行"""
    if not text:
        return ""
    # 将 <br> 替换为换行符
    text = _BR_RE.sub("\n", text)
    # 去除所有 HTML 标签
    text = _TAG_RE.sub("", text)
    return text.strip()