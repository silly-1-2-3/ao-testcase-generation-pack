import os
import subprocess
import sys

os.environ["http_proxy"] = "http://127.0.0.1:1234"
os.environ["https_proxy"] = "http://127.0.0.1:1234"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# ==========================================
# 组装 vLLM 的启动命令
# ==========================================
command = [
    sys.executable,  # 自动获取当前虚拟环境的 python 解释器路径
    "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen2.5-7B-Instruct",
    "--served-model-name", "qwen2.5",
    "--max-model-len", "8192",
    "--download-dir", "/home/tongyufei/Qwen/Qwen2.5-7B-instruct",
    "--gpu-memory-utilization", "0.80",
    "--enable-auto-tool-choice",      # 允许服务器自动处理工具调用逻辑
    "--tool-call-parser", "hermes"    # 明确告诉 vLLM：请使用 Qwen2.5 专用的 XML 标签解析规则！
]

print("🚀 准备启动 vLLM API 服务器...")
print("💻 执行命令:", " ".join(command))
print("-" * 50)

# ==========================================
# 3. 运行服务并接管输出
# ==========================================
try:
    # 启动子进程，这个函数会一直阻塞在这里（保持服务器运行），
    # 直到你按下 Ctrl+C 或者程序崩溃
    subprocess.run(command, check=True)
except KeyboardInterrupt:
    print("\n🛑 收到中断信号，vLLM 服务器已安全关闭。")
except subprocess.CalledProcessError as e:
    print(f"\n❌ 启动失败，请检查报错信息。错误码: {e.returncode}")