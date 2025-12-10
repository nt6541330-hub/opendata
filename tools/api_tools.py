import asyncio
from fastapi import APIRouter
from common.utils import get_logger

# 引入各个子工具模块
from tools.text_annotation import router as annotation_router
from tools.text_annotation import warmup as annotation_warmup

from tools.word_cloud import router as word_cloud_router
from tools.word_cloud import warmup as word_cloud_warmup

# 【新增】引入图像描述模块
from tools.image_description import router as img_desc_router
from tools.image_description import warmup as img_desc_warmup

# 抓取引擎
from tools.capture_engine import router as capture_router

logger = get_logger(__name__)

# 创建总路由
router = APIRouter()

# 注册子路由
router.include_router(annotation_router, prefix="/annotation", tags=["工具-文本标注"])
router.include_router(word_cloud_router, prefix="/wordcloud", tags=["工具-热点词云"])
router.include_router(capture_router, prefix="/capture", tags=["工具-抓取引擎"])

# 【新增】注册图像描述路由 -> /tools/image_desc/analyze
router.include_router(img_desc_router, prefix="/image_desc", tags=["工具-图像描述"])

# 统一预热入口
async def warmup_all_tools():
    """启动所有工具的预热任务"""
    logger.info("正在初始化工具集预热任务...")

    tasks = [
        annotation_warmup(),
        word_cloud_warmup(),
        img_desc_warmup()  # 【新增】加入图像模型预热
    ]

    # 并发执行所有预热
    if tasks:
        await asyncio.gather(*tasks)