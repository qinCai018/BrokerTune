#!/bin/bash

# BrokerTuner - 拷贝Mosquitto配置文件
# 从系统配置目录拷贝配置文件到本地目录

set -e  # 遇到错误立即退出

echo "=========================================="
echo "BrokerTuner - 拷贝Mosquitto配置文件"
echo "=========================================="

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_CONFIG="/etc/mosquitto/mosquitto.conf"
TARGET_CONFIG="$SCRIPT_DIR/mosquitto.conf"

echo "脚本目录: $SCRIPT_DIR"
echo "源配置文件: $SOURCE_CONFIG"
echo "目标配置文件: $TARGET_CONFIG"
echo ""

# 检查源配置文件是否存在
if [ ! -f "$SOURCE_CONFIG" ]; then
    echo "错误: 源配置文件不存在: $SOURCE_CONFIG"
    echo "请确保Mosquitto已正确安装"
    exit 1
fi

# 备份现有配置文件（如果存在）
if [ -f "$TARGET_CONFIG" ]; then
    BACKUP_FILE="${TARGET_CONFIG}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "备份现有配置文件到: $BACKUP_FILE"
    cp "$TARGET_CONFIG" "$BACKUP_FILE"
fi

# 拷贝配置文件
echo "正在拷贝配置文件..."
sudo cp "$SOURCE_CONFIG" "$TARGET_CONFIG"

# 确保文件权限正确
sudo chown $(whoami):$(whoami) "$TARGET_CONFIG"
chmod 644 "$TARGET_CONFIG"

echo "✅ 配置文件已成功拷贝到: $TARGET_CONFIG"
echo ""
echo "配置文件内容预览（前20行）:"
head -20 "$TARGET_CONFIG"

echo ""
echo "=========================================="
