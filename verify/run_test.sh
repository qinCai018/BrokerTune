#!/bin/bash
# 运行吞吐量测试脚本

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "吞吐量测试"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "throughput_test.py" ]; then
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
    # 尝试自动检测
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

if [ -z "$EMQTT_BENCH_PATH" ]; then
    echo "警告: 未找到emqtt_bench，请设置EMQTT_BENCH_PATH环境变量"
    echo "  export EMQTT_BENCH_PATH=/path/to/emqtt_bench"
fi

# 设置PYTHONPATH
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

echo "开始运行测试..."
echo ""

# 运行测试脚本
sudo -E python3 throughput_test.py "$@"

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "结果文件: throughput_test_results.csv"
echo ""
