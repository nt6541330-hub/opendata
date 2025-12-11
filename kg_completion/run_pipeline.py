import subprocess
import sys
import os


def run_script(name):
    print(f"\n{'=' * 10} Running {name} {'=' * 10}")
    script_path = os.path.join(os.path.dirname(__file__), name)
    ret = subprocess.call([sys.executable, script_path])
    if ret != 0:
        print(f"❌ {name} 执行失败！退出。")
        sys.exit(ret)


if __name__ == "__main__":
    # 1. 导出数据
    run_script("step1_export_data.py")

    # 2. 训练图嵌入
    run_script("step2_train_transe.py")

    # 3. 微调 LLM (Output: LoRA weights only)
    run_script("step3_finetune_llm.py")

    # 4. 推理 (Base Model + LoRA Adapter)
    run_script("step4_inference.py")

    print("\n✅ 全流程执行成功！")