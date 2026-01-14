#!/bin/bash

# BrokerTuner - 运行性能指标收集脚本
# 便捷的 bash 脚本包装器

set -e  # 遇到错误立即退出

echo "=========================================="
echo "BrokerTuner - 性能指标收集"
echo "=========================================="

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT_SCRIPT="$SCRIPT_DIR/collect_metrics.py"

# 检查 Python 脚本是否存在
if [ ! -f "$COLLECT_SCRIPT" ]; then
    echo "错误: 收集脚本不存在: $COLLECT_SCRIPT"
    exit 1
fi

# 检查 Python 是否可用
if ! command -v python3 &> /dev/null; then
    echo "错误: Python3 未安装或不在 PATH 中"
    exit 1
fi

# 检查 Broker 是否运行
echo "检查 Broker 连接..."
if ! timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/1883" 2>/dev/null; then
    echo "⚠️  警告: 无法连接到 Broker (127.0.0.1:1883)"
    echo "请确保 Broker 正在运行"
    echo ""
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "运行性能指标收集脚本..."
echo ""

# 运行 Python 脚本，传递所有参数
python3 "$COLLECT_SCRIPT" "$@"

echo ""
echo "=========================================="
