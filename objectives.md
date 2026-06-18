**项目描述文本：**

本项目以 **vibe-coding** 方式，从零开始搭建一套完整的强化学习（RL）训练框架，目标是训练一个能在经典坦克大战（Battle City）游戏中实现“一命通关”（one-life clear）的智能体（Agent）。这是一个教学导向的项目，通过详细的代码实现和丰富中文注释，系统讲解RL核心原理，包括MDP、马尔可夫决策过程、Bellman方程、Policy Gradient、Advantage估计、PPO的Clip机制、熵正则化等深层概念，帮助大家轻松入坑RL。

整个框架基于Python实现，将选用一个开源的Pygame坦克大战游戏克隆作为基础环境（推荐参考 tirinox/pybattlecity 等稳定仓库），包装成标准的Gymnasium Env，支持rgb_array渲染模式以便实时观察AI玩游戏并通过RecordVideo生成训练过程视频（直观展示Agent从随机行为到熟练通关的进化）。优先采用tile-based简化状态快速验证，后续升级到像素输入+CNN；支持reward shaping、curriculum learning和并行采样以提升训练稳定性。

**硬件与环境要求**：本地使用一张RTX 4060 8GB显卡，必要时可考虑租用云平台加速大规模训练。所有依赖置于一个全新的Conda环境 `tank-rl-teach` 中，选用稳定可靠的 **Python 3.11** 版本。

**核心依赖**（在conda环境内通过pip安装）：
```bash
conda create -n tank-rl-teach python=3.11
conda activate tank-rl-teach
```

项目强调可复现性、详细文档和可视化（TensorBoard/Matplotlib曲线 + 游戏视频），适合制作教程或分享。
