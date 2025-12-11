import os
import sys
import json
import shutil
import subprocess
from kg_config import PROJECT_ROOT, LLAMA_FACTORY_DIR, LLM_DIR, LLM_BASE_MODEL_PATH, LORA_OUTPUT_DIR


def main():
    print(">>> [Step 3] 开始 LLM LoRA 微调...")

    # 1. 注册数据集到 LLaMA-Factory
    data_src = os.path.join(LLM_DIR, "kgc_train.json")
    data_dst = os.path.join(LLAMA_FACTORY_DIR, "data", "kgc_train.json")
    shutil.copy(data_src, data_dst)

    info_file = os.path.join(LLAMA_FACTORY_DIR, "data", "dataset_info.json")
    with open(info_file, 'r') as f:
        info = json.load(f)

    info["kgc_train"] = {
        "file_name": "kgc_train.json",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"}
    }

    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)

    # 2. 构建训练命令
    # 注意: output_dir 只会包含 adapter_model.bin/safetensors，不会包含底座模型
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

    print(f"    执行命令: {' '.join(cmd)}")
    print("    正在训练... (日志将直接打印在下方)")

    # 使用 subprocess 调用
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    ret = subprocess.call(cmd, cwd=LLAMA_FACTORY_DIR, env=env)

    if ret != 0:
        raise RuntimeError("LLM 训练失败")

    print(f">>> [Step 3] 微调完成。LoRA 权重已保存至: {LORA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()