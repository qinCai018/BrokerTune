#!/bin/bash

# BrokerTuner - 启动Broker脚本
# 使用本地配置文件启动Mosquitto Broker

set -e  # 遇到错误立即退出

echo "=========================================="
echo "BrokerTuner - 启动Broker"
echo "=========================================="

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 优先使用 broker_tuner.conf（完整的独立配置文件）
# 如果不存在，则使用 mosquitto.conf（从系统拷贝的配置文件）
if [ -f "$SCRIPT_DIR/broker_tuner.conf" ]; then
    CONFIG_FILE="$SCRIPT_DIR/broker_tuner.conf"
elif [ -f "$SCRIPT_DIR/mosquitto.conf" ]; then
    CONFIG_FILE="$SCRIPT_DIR/mosquitto.conf"
else
    echo "错误: 未找到配置文件"
    echo "请确保存在以下文件之一:"
    echo "  - $SCRIPT_DIR/broker_tuner.conf (推荐，完整独立配置)"
    echo "  - $SCRIPT_DIR/mosquitto.conf (从系统拷贝)"
    exit 1
fi

echo "脚本目录: $SCRIPT_DIR"
echo "配置文件: $CONFIG_FILE"
echo ""

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查Mosquitto是否已安装
if ! command -v mosquitto &> /dev/null; then
    echo "错误: Mosquitto未安装或不在PATH中"
    echo "请安装Mosquitto: sudo apt install mosquitto"
    exit 1
fi

echo "1. 停止现有Mosquitto服务..."
sudo systemctl stop mosquitto || true
sudo pkill -f mosquitto || true

# 等待端口释放
echo "2. 等待端口1883释放..."
for i in {1..30}; do
    if ! netstat -tln 2>/dev/null | grep -q ":1883 "; then
        break
    fi
    echo "   等待中... ($i/30)"
    sleep 1
done

echo "3. 启动Mosquitto Broker（使用本地配置）..."
echo "   命令: mosquitto -c $CONFIG_FILE -v"
echo ""

# 使用本地配置文件启动Mosquitto（前台运行）
# -c 指定配置文件
# -v 启用详细日志
mosquitto -c "$CONFIG_FILE" -v

echo "Broker已停止"