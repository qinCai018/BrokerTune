#!/bin/bash
# 运行简化的吞吐量测试脚本

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "简化吞吐量测试"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "simple_throughput_test.py" ]; then
    echo "错误: 请在verify目录运行此脚本"
    exit 1
fi

# 检查Mosquitto是否运行
if ! systemctl is-active --quiet mosquitto; then
    echo "错误: Mosquitto服务未运行"
    echo "请先启动Mosquitto: sudo systemctl start mosquitto"
    exit 1
fi

# 检查emqtt_bench是否可用
if [ -z "$EMQTT_BENCH_PATH" ]; then
    POSSIBLE_PATHS=(
        "/home/qincai/userDir/mosquitto/BrokerPressTest/tool/emqtt_bench_release/bin/emqtt_bench"
        "$(which emqtt_bench 2>/dev/null)"
    )
    
    for path in "${POSSIBLE_PATHS[@]}"; do
        if [ -n "$path" ] && [ -f "$path" ] && [ -x "$path" ]; then
            export EMQTT_BENCH_PATH="$path"
            echo "自动检测到 emqtt_bench: $EMQTT_BENCH_PATH"
            break
        fi
    done
fi

# 设置PYTHONPATH
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

echo "测试内容:"
echo "  配置1: 默认配置（所有参数默认）"
echo "  配置2: max_inflight_messages=100，其他参数默认"
echo "  工作负载: 1024B, QoS=1, 10ms"
echo ""
echo "开始运行测试..."
echo ""

# 运行测试脚本
sudo env PYTHONPATH="$PYTHONPATH" HOME="$HOME" SUDO_USER="${SUDO_USER:-$USER}" EMQTT_BENCH_PATH="${EMQTT_BENCH_PATH:-}" python3 simple_throughput_test.py "$@"

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
