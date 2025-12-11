# open_source_data/main.py
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from config.settings import settings
from data_collect.api_data_collect import router as collect_router
from tools.api_tools import router as tools_router
from tools.api_tools import warmup_all_tools

# 引入 Semantic 路由
from semantic.api import router as semantic_router
# 引入 pipeline 实例用于启动时自动运行
from semantic.pipeline import pipeline_instance


# 使用 lifespan 替代 on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 工具预热
    asyncio.create_task(warmup_all_tools())

    # 2. 【可选】自动启动语义处理监控 (如果不希望自动启动，可注释掉下面这一行)
    asyncio.create_task(pipeline_instance.run_loop(interval=60))

    yield
    # 3. 关闭时的清理工作 (如有)
    pipeline_instance.stop()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="多源数据采集与智能处理平台",
    version="1.3.0",
    lifespan=lifespan  # 注册 lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(collect_router, prefix="/collect", tags=["数据采集"])
app.include_router(tools_router, prefix="/tools")
app.include_router(semantic_router, prefix="/semantic", tags=["语义处理"])


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "project": settings.PROJECT_NAME,
        "modules": ["collection", "tools", "semantic"]
    }


# 旧的 on_event 方式已移除

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8801,
        reload=False,
        reload_excludes=["logs", "logs/*", "*.log"]
    )