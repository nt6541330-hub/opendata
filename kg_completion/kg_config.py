import os
import sys

# 将项目根目录加入路径，确保能读到 config.settings
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

try:
    from config.settings import settings
except ImportError:
    print("警告: 未找到 config.settings，使用默认配置")
    class MockSettings:
        NEBULA_IP = "39.104.200.88"
        NEBULA_PORT = 41003
        NEBULA_USER = "root"
        NEBULA_PASSWORD = "123456"
        NEBULA_SPACE_NAME = "event_target1"
    settings = MockSettings()

# === 路径规划 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "kgc_data")
TRANSE_DIR = os.path.join(DATA_DIR, "transe")
LLM_DIR = os.path.join(DATA_DIR, "llm")

# 模型相关路径
LLM_BASE_MODEL_PATH = "/mnt/data/Qwen3-8B"  # 底座模型路径 (只读)
LORA_OUTPUT_DIR = "/open_source_data/checkpoints/qwen_kgc_lora" # LoRA 权重保存路径
TRANSE_VECTORS_PATH = os.path.join(PROJECT_ROOT, "transe_vectors.json") # TransE 向量文件

# LLaMA-Factory 路径
LLAMA_FACTORY_DIR = os.path.join(PROJECT_ROOT, "LLaMA-Factory")

# === Nebula 连接配置 ===
NEBULA_CONFIG = {
    "hosts": [(settings.NEBULA_IP, settings.NEBULA_PORT)],
    "user": settings.NEBULA_USER,
    "password": settings.NEBULA_PASSWORD,
    "space": settings.NEBULA_SPACE_NAME
}

# 自动创建目录
for d in [DATA_DIR, TRANSE_DIR, LLM_DIR, os.path.dirname(LORA_OUTPUT_DIR)]:
    os.makedirs(d, exist_ok=True)