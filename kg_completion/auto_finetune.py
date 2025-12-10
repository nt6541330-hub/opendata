import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ================= 配置区域 (请根据实际情况修改) =================

# 1. 路径配置
PROJECT_ROOT = "/open_source_data/kg_completion"
LLAMA_FACTORY_DIR = os.path.join(PROJECT_ROOT, "LLaMA-Factory")
SOURCE_DATA_PATH = os.path.join(PROJECT_ROOT, "kgc_data/llm/kgc_train.json")  # 之前导出脚本生成的路径
BASE_MODEL_PATH = "/mnt/data/Qwen3-8B"
OUTPUT_DIR = "/open_source_data/checkpoints/qwen_kgc_lora"

# 2. 训练超参数配置
TRAIN_ARGS = {
    "stage": "sft",
    "do_train": True,
    "model_name_or_path": BASE_MODEL_PATH,
    "dataset": "kgc_train",  # 注册在 dataset_info.json 中的名字
    "template": "qwen",  # 对应 Qwen 模型
    "finetuning_type": "lora",
    "lora_target": "all",
    "output_dir": OUTPUT_DIR,
    "overwrite_output_dir": True,
    "cutoff_len": 2048,  # 处理长文本属性
    "preprocessing_num_workers": 1,  # 【关键】解决 Python 3.11 多进程报错
    "per_device_train_batch_size": 2,  # 显存优化
    "gradient_accumulation_steps": 8,  # 梯度累积
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "warmup_ratio": 0.1,
    "save_steps": 100,
    "learning_rate": 1e-4,
    "num_train_epochs": 3.0,
    "bf16": True,  # 开启 bf16 加速
    "plot_loss": True
}


# ================= 核心逻辑 =================

def check_paths():
    """检查必要的文件和目录是否存在"""
    print(">>> [1/4] 环境检查...")
    if not os.path.exists(LLAMA_FACTORY_DIR):
        raise FileNotFoundError(f"未找到 LLaMA-Factory 目录: {LLAMA_FACTORY_DIR}")

    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(f"未找到训练数据: {SOURCE_DATA_PATH}，请先运行导出脚本。")

    print("    环境检查通过。")


def prepare_dataset():
    """复制数据并注册到 dataset_info.json"""
    print(">>> [2/4] 准备数据集...")

    # 1. 复制文件到 LLaMA-Factory/data 目录
    target_data_dir = os.path.join(LLAMA_FACTORY_DIR, "data")
    target_file = os.path.join(target_data_dir, "kgc_train.json")

    try:
        shutil.copy(SOURCE_DATA_PATH, target_file)
        print(f"    数据已复制到: {target_file}")
    except Exception as e:
        raise RuntimeError(f"复制数据失败: {e}")

    # 2. 修改 dataset_info.json
    info_file = os.path.join(target_data_dir, "dataset_info.json")

    if not os.path.exists(info_file):
        raise FileNotFoundError(f"未找到 dataset_info.json: {info_file}")

    try:
        with open(info_file, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)

        # 注册数据集配置
        dataset_info["kgc_train"] = {
            "file_name": "kgc_train.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }

        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        print("    dataset_info.json 注册更新成功。")

    except Exception as e:
        raise RuntimeError(f"更新 dataset_info.json 失败: {e}")


def run_training():
    """构建命令并执行训练"""
    print(f">>> [3/4] 开始训练任务...")
    print(f"    基座模型: {BASE_MODEL_PATH}")
    print(f"    输出目录: {OUTPUT_DIR}")

    # 构建命令行参数列表
    cmd = [sys.executable, "-m", "llamafactory.cli", "train"]


    for key, value in TRAIN_ARGS.items():
        # 布尔值参数处理：True 则添加 flag，False 则忽略（或根据特定逻辑）
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    # 打印完整命令供调试
    print("\nExecuting Command:")
    print(" ".join(cmd))
    print("-" * 50)

    # 切换工作目录到 LLaMA-Factory 并执行
    try:
        # 使用 CUDA_VISIBLE_DEVICES=0 环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        # 实时流式输出日志
        process = subprocess.Popen(
            cmd,
            cwd=LLAMA_FACTORY_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # 实时打印输出
        for line in process.stdout:
            print(line, end='')

        process.wait()

        if process.returncode == 0:
            print("\n>>> [4/4] 训练成功完成！")
            print(f"模型权重已保存至: {OUTPUT_DIR}")
        else:
            print("\n>>> 错误: 训练进程异常退出。")
            sys.exit(process.returncode)

    except KeyboardInterrupt:
        print("\n>>> 用户手动停止训练。")
    except Exception as e:
        print(f"\n>>> 执行训练时发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        check_paths()
        prepare_dataset()
        run_training()
    except Exception as e:
        print(f"\n[ERROR] 脚本执行失败: {e}")
        sys.exit(1)