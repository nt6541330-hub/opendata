import os
import sys
import json
import shutil
import subprocess
from kg_config import PROJECT_ROOT, LLAMA_FACTORY_DIR, LLM_DIR, LLM_BASE_MODEL_PATH, LORA_OUTPUT_DIR


def main():
    print(">>> [Step 3] 开始 LLM LoRA 微调...")

    # === 1. 更加健壮的数据注册流程 ===
    data_src = os.path.join(LLM_DIR, "kgc_train.json")

    # 目标目录：LLaMA-Factory/data
    target_data_dir = os.path.join(LLAMA_FACTORY_DIR, "data")
    data_dst = os.path.join(target_data_dir, "kgc_train.json")

    print(f"    源文件: {data_src}")
    print(f"    目标地: {data_dst}")

    # 【关键修正】强制创建目标目录，防止 "FileNotFoundError"
    if not os.path.exists(target_data_dir):
        print(f"    警告: 目录 {target_data_dir} 不存在，正在自动创建...")
        os.makedirs(target_data_dir, exist_ok=True)

    # 复制文件
    try:
        shutil.copy(data_src, data_dst)
        print("    数据复制成功。")
    except Exception as e:
        print(f"    ❌ 数据复制失败: {e}")
        # 如果复制失败，打印当前目录结构帮助调试
        print(
            f"    当前 {LLAMA_FACTORY_DIR} 下的文件: {os.listdir(LLAMA_FACTORY_DIR) if os.path.exists(LLAMA_FACTORY_DIR) else '目录不存在'}")
        sys.exit(1)

    # 更新 dataset_info.json
    info_file = os.path.join(target_data_dir, "dataset_info.json")

    # 如果 info 文件不存在（比如刚 clone 下来），初始化一个空的
    if not os.path.exists(info_file):
        print("    dataset_info.json 不存在，初始化一个新的...")
        info = {}
    else:
        with open(info_file, 'r', encoding='utf-8') as f:
            try:
                info = json.load(f)
            except json.JSONDecodeError:
                info = {}  # 文件损坏则重置

    # 注册数据集
    info["kgc_train"] = {
        "file_name": "kgc_train.json",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"}
    }

    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("    dataset_info.json 注册完成。")

    # === 2. 构建训练命令 ===
    # 检查 adapter_config.json 是否存在，防止重复训练（可选）
    if os.path.exists(os.path.join(LORA_OUTPUT_DIR, "adapter_config.json")):
        print("    检测到 LoRA 权重已存在，跳过训练 (如需重训请删除 checkpoints 目录)")
        return

    cmd = [
        sys.executable, "-m", "llamafactory.cli", "train",
        "--stage", "sft",
        "--do_train",
        "--model_name_or_path", LLM_BASE_MODEL_PATH,
        "--dataset", "kgc_train",
        "--template", "qwen",
        "--finetuning_type", "lora",
        "--lora_target", "all",
        "--output_dir", LORA_OUTPUT_DIR,
        "--overwrite_output_dir",
        "--per_device_train_batch_size", "4",
        "--gradient_accumulation_steps", "4",
        "--lr_scheduler_type", "cosine",
        "--learning_rate", "1e-4",
        "--num_train_epochs", "3.0",
        "--bf16"
    ]

    print(f"    执行训练命令...")

    # 使用 subprocess 调用
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        ret = subprocess.call(cmd, cwd=LLAMA_FACTORY_DIR, env=env)
        if ret != 0:
            print("    ❌ 训练进程返回错误代码。请检查上方日志。")
            sys.exit(ret)
    except FileNotFoundError:
        print(f"    ❌ 无法在 {LLAMA_FACTORY_DIR} 下找到 llamafactory 模块。")
        print("    请确保你已在该目录下运行过: pip install -e .")
        sys.exit(1)

    print(f">>> [Step 3] 微调完成。LoRA 权重已保存至: {LORA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()