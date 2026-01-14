#!/bin/bash
# 启动吞吐量测试（带进度条显示）

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "启动吞吐量测试（带进度条）"
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

# 检查tqdm是否安装
if ! python3 -c "import tqdm" 2>/dev/null; then
    echo "提示: tqdm未安装，将使用文本进度显示"
    echo "安装命令: pip install tqdm"
    echo ""
fi

# 设置PYTHONPATH（确保包含项目根目录）
PROJECT_ROOT="$(cd .. && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "测试配置:"
echo "  - Broker配置: 2种（max_inflight_100, default）"
echo "  - 工作负载组合: 12种"
echo "  - 总测试用例: 24个"
echo "  - 预计时间: 约27-30分钟"
echo ""
echo "PYTHONPATH: $PYTHONPATH"
echo "项目根目录: $PROJECT_ROOT"
echo ""
echo "开始运行测试（带进度条）..."
echo ""

# 运行测试脚本
# 使用与run_train.sh相同的方式设置PYTHONPATH
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 切换到verify目录运行脚本
cd verify

# 确保PYTHONPATH和其他必要的环境变量传递给sudo
# 保留SUDO_USER以便Python脚本能找到原始用户的site-packages
# 传递EMQTT_BENCH_PATH以便WorkloadManager能找到emqtt_bench
sudo env PYTHONPATH="$PYTHONPATH" HOME="$HOME" SUDO_USER="${SUDO_USER:-$USER}" EMQTT_BENCH_PATH="${EMQTT_BENCH_PATH:-}" python3 throughput_test.py "$@"

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "结果文件: throughput_test_results.csv"
echo ""
