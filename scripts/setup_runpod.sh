#!/bin/bash
# ============================================================
# RunPod 4090 训练环境一键配置脚本
# ============================================================
# 用法:
#   1. 在 RunPod 上启动一个 PyTorch 模板的 Pod (4090)
#   2. 将项目代码上传到 /workspace/
#   3. 运行: bash scripts/setup_runpod.sh
# ============================================================
set -e

echo "=== RunPod Battle City RL 环境配置 ==="

# 检查 CUDA 是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "[错误] 未检测到 nvidia-smi, 请确认使用的是 GPU Pod"
    exit 1
fi

echo "[信息] GPU 信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 初始化 git 子模块
echo "[步骤 1/4] 初始化 git 子模块..."
git submodule update --init --recursive

# 创建 conda 环境 (如果 RunPod 模板已预装 conda)
echo "[步骤 2/4] 创建 conda 环境..."
if command -v conda &> /dev/null; then
    conda create -n tank-rl-runpod python=3.11 -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate tank-rl-runpod
else
    echo "[警告] 未找到 conda, 使用系统 Python"
    python3 -m venv /workspace/venv
    source /workspace/venv/bin/activate
fi

# 安装 PyTorch CUDA 版
echo "[步骤 3/4] 安装 PyTorch (CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装其余依赖
echo "[步骤 4/4] 安装项目依赖..."
pip install gymnasium numpy tensorboard matplotlib pygame moviepy pytest opencv-python

# 验证安装
echo ""
echo "=== 验证安装 ==="
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.__version__}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "=== 配置完成 ==="
echo "启动训练:"
echo "  conda activate tank-rl-runpod"
echo "  python scripts/train_tir.py --device cuda --run-name runpod_exp1"
echo ""
echo "使用 nohup 后台训练:"
echo "  nohup python scripts/train_tir.py --device cuda --run-name runpod_exp1 > train.log 2>&1 &"
echo "查看日志: tail -f train.log"
