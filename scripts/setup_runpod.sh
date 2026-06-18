#!/bin/bash
# ============================================================
# RunPod 4090 训练环境一键配置脚本
# ============================================================
# 前置条件:
#   1. RunPod 上启动 PyTorch 模板 Pod (RTX 4090)
#   2. 创建 Network Volume 并挂载到 /workspace
#   3. 通过 Web Terminal 或 SSH 登录 Pod
#
# 用法:
#   git clone --recurse-submodules -b mac \
#       https://github.com/JYuDanny/battle-city-rl-scratch.git \
#       /workspace/battle-city-rl
#   cd /workspace/battle-city-rl
#   bash scripts/setup_runpod.sh
# ============================================================
set -e

NETWORK_VOLUME_DIR="/workspace"

echo "============================================"
echo " Battle City RL — RunPod 环境配置"
echo "============================================"

# ── 0. 检查 CUDA ──────────────────────────────────────────
echo ""
echo "[0/5] 检查 CUDA 环境..."

if ! command -v nvidia-smi &> /dev/null; then
    echo "[错误] 未检测到 nvidia-smi, 请确认使用 GPU Pod (非 CPU Pod)"
    exit 1
fi

echo "  GPU 信息:"
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version \
    --format=csv,noheader 2>/dev/null | sed 's/^/    /'

# ── 1. Git 子模块 ─────────────────────────────────────────
echo ""
echo "[1/5] 初始化 git 子模块..."
git submodule update --init --recursive
echo "  完成"

# ── 2. Conda 环境 ─────────────────────────────────────────
echo ""
echo "[2/5] 创建 conda 环境 (tank-rl-runpod)..."

if command -v conda &> /dev/null; then
    # 检查环境是否已存在
    if conda env list | grep -q "^tank-rl-runpod "; then
        echo "  环境已存在, 跳过创建"
    else
        conda create -n tank-rl-runpod python=3.11 -y
        echo "  创建完成"
    fi
else
    echo "[警告] 未找到 conda, 尝试使用系统 Python + venv"
    python3 -m venv /workspace/venv-tank-rl
    echo "  source /workspace/venv-tank-rl/bin/activate" > /workspace/activate.sh
    echo "  已创建虚拟环境, 激活命令已写入 /workspace/activate.sh"
fi

# ── 3. PyTorch CUDA ───────────────────────────────────────
echo ""
echo "[3/5] 安装 PyTorch (CUDA 12.4)..."
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate tank-rl-runpod
fi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo "  完成"

# ── 4. 项目依赖 ──────────────────────────────────────────
echo ""
echo "[4/5] 安装项目依赖..."
pip install -e .
pip install opencv-python
echo "  完成"

# ── 5. 持久化存储 (Network Volume) ────────────────────────
echo ""
echo "[5/5] 配置持久化存储..."

# 检测 Network Volume 挂载点 (RunPod 默认挂载到 /workspace)
PERSIST_BASE=""
for candidate in "$NETWORK_VOLUME_DIR/checkpoints" "$NETWORK_VOLUME_DIR/persist" "$HOME/persist"; do
    if [ -d "$candidate" ]; then
        PERSIST_BASE="$candidate"
        break
    fi
done

if [ -z "$PERSIST_BASE" ]; then
    # 未检测到 Network Volume, 使用本地目录 (Pod 停止后数据丢失!)
    echo "[警告] 未检测到 Network Volume 挂载点!"
    echo "  checkpoint 将保存到本地磁盘 (Pod 停止后丢失)"
    echo "  建议创建 Network Volume 并重新挂载 Pod"
    mkdir -p checkpoints runs
else
    echo "  检测到持久化目录: $PERSIST_BASE"
    # 创建符号链接, 让训练脚本自动写入 Network Volume
    rm -rf checkpoints runs 2>/dev/null || true
    mkdir -p "$PERSIST_BASE/checkpoints" "$PERSIST_BASE/runs"
    ln -sfn "$PERSIST_BASE/checkpoints" checkpoints
    ln -sfn "$PERSIST_BASE/runs" runs
    echo "  符号链接已创建:"
    echo "    checkpoints → $PERSIST_BASE/checkpoints"
    echo "    runs       → $PERSIST_BASE/runs"
fi

# ── 验证 ──────────────────────────────────────────────────
echo ""
echo "============================================"
echo " 验证安装"
echo "============================================"

python -c "
import torch
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA 可用:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:        {torch.cuda.get_device_name(0)}')
    mem_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'  VRAM:       {mem_gb:.1f} GB')
print(f'  MPS 可用:   {torch.backends.mps.is_available()}')
"

# 运行测试
echo ""
echo " 运行 pytest..."
pytest tests/ -q 2>&1 | tail -3

echo ""
echo "============================================"
echo " 配置完成! 启动训练:"
echo ""
echo "  # 激活环境"
echo "  conda activate tank-rl-runpod"
echo ""
echo "  # 最小化测试 (10240 步, 约 15 分钟)"
echo "  python scripts/train_tir.py --device cuda --total-steps 10240 --run-name test"
echo ""
echo "  # 正式训练 (5M 步, 约 2-3 天, 建议 nohup 后台运行)"
echo "  nohup python scripts/train_tir.py --device cuda --run-name runpod_exp1 \\"
echo "      > train.log 2>&1 &"
echo ""
echo "  # 查看日志"
echo "  tail -f train.log"
echo ""
echo "  # 查看 GPU 使用情况"
echo "  watch -n 2 nvidia-smi"
echo "============================================"
