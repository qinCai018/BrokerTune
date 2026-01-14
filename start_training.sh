#!/bin/bash
# 强化学习训练启动脚本
# 使用方法: ./start_training.sh [参数]

set -e

# 切换到项目根目录
cd "$(dirname "$0")"

echo "=========================================="
echo "启动强化学习训练"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "tuner/train.py" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 默认参数
TOTAL_TIMESTEPS=${1:-100000}
SAVE_DIR=${2:-./checkpoints}
SAVE_FREQ=${3:-10000}

echo "训练参数:"
echo "  - 总步数: $TOTAL_TIMESTEPS"
echo "  - 保存目录: $SAVE_DIR"
echo "  - 保存频率: $SAVE_FREQ"
echo ""

# 运行训练脚本
./script/run_train.sh \
    --enable-workload \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --save-dir "$SAVE_DIR" \
    --save-freq "$SAVE_FREQ" \
    --workload-publishers 100 \
    --workload-subscribers 10 \
    --workload-publisher-interval-ms 15 \
    --workload-message-size 512 \
    --workload-qos 1 \
    --workload-topic "test/topic"

echo ""
echo "训练完成！"
