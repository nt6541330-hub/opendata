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

# 【新增】引入 KG Completion 模块
from kg_completion.api_server import router as kg_router, init_kg_service, stop_kg_service

# ================= 新增: 引入 KGE 模块 =================
from kg_correction.kge import router as kg_correction_router
from kg_correction.kge import corrector
# ======================================================


# 使用 lifespan 替代 on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 工具预热
    asyncio.create_task(warmup_all_tools())

    # 2. 【可选】自动启动语义处理监控
    # asyncio.create_task(pipeline_instance.run_loop(interval=60))

    # 3. 【新增】KG 服务初始化 (加载大模型和图谱向量)
    # 注意：这会加载模型，可能需要一些时间
    await init_kg_service()

    # 2. 【新增】加载知识图谱纠错大模型
    # 注意：加载 8B 模型需要显存，如果显存紧张建议检查 kge.py 中的加载配置
    corrector.load_model()

    yield

    # 4. 关闭时的清理工作
    pipeline_instance.stop()

    # 5. 【新增】KG 服务清理
    stop_kg_service()


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
# 【新增】注册 KG 路由
app.include_router(kg_router, prefix="/kgc", tags=["知识图谱补全"])

# ================= 新增: 注册 KGE 路由 =================
app.include_router(kg_correction_router, prefix="/kge", tags=["知识图谱纠错"])
# ======================================================


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "project": settings.PROJECT_NAME,
        "modules": ["collection", "tools", "semantic", "kg_completion"]
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8801,
        reload=False,
        reload_excludes=["logs", "logs/*", "*.log"]
    )