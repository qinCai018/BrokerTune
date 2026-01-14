#!/bin/bash
# 清理训练产生的中间文件和训练结果

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "清理训练文件和结果"
echo "=========================================="
echo ""

# 计算清理前的空间
BEFORE=$(du -sh checkpoints/ 2>/dev/null | cut -f1 || echo "0")
echo "清理前checkpoints目录大小: $BEFORE"
echo ""

# 删除模型文件
echo "删除模型文件..."
rm -f checkpoints/ddpg_mosquitto_*.zip
echo "✅ 模型文件已删除"

# 删除replay buffer文件
echo ""
echo "删除replay buffer文件..."
rm -f checkpoints/*replay_buffer*.pkl
echo "✅ Replay buffer文件已删除"

# 删除训练日志
echo ""
echo "删除训练日志..."
rm -f checkpoints/action_throughput_log.csv
rm -rf checkpoints/logs/*
echo "✅ 训练日志已删除"

# 删除monitor日志
echo ""
echo "删除monitor日志..."
rm -rf checkpoints/monitor/*
echo "✅ Monitor日志已删除"

# 计算清理后的空间
AFTER=$(du -sh checkpoints/ 2>/dev/null | cut -f1 || echo "0")
echo ""
echo "=========================================="
echo "清理完成！"
echo "=========================================="
echo "清理前: $BEFORE"
echo "清理后: $AFTER"
echo ""
echo "当前磁盘使用情况:"
df -h / | tail -1
