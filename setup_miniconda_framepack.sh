#!/bin/bash

# 设置安装位置
INSTALL_DIR="$HOME/miniconda3"
ENV_NAME="framepack"
PYTHON_VERSION="3.10"

# 下载Miniconda安装脚本
CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
TMP_INSTALLER="/tmp/miniconda.sh"

echo "下载Miniconda安装程序..."
wget "$CONDA_URL" -O "$TMP_INSTALLER"

echo "静默安装Miniconda到 $INSTALL_DIR..."
bash "$TMP_INSTALLER" -b -p "$INSTALL_DIR"

echo "删除安装脚本..."
rm "$TMP_INSTALLER"

echo "初始化conda..."
source "$INSTALL_DIR/bin/activate"
conda init bash

# 激活conda
source ~/.bashrc

echo "创建名为$ENV_NAME的Python环境 (Python=$PYTHON_VERSION)..."
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

echo "激活环境并安装PyTorch、torchvision和torchaudio..."
source "$INSTALL_DIR/bin/activate" "$ENV_NAME"
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 如果你不需要CUDA，使用以下命令替代上面的安装：
# conda install -y pytorch torchvision torchaudio cpuonly -c pytorch

# 安装额外指定的依赖包
pip install diffusers datasets wandb scikit-video tqdm

# 检查并安装requirements.txt
if [ -f "requirements.txt" ]; then
    echo "安装 requirements.txt 中指定的依赖包..."
    pip install -r requirements.txt
else
    echo "当前目录未找到 requirements.txt，请确认后再手动运行 pip install -r requirements.txt"
fi

echo "环境 $ENV_NAME 已成功设置！"
