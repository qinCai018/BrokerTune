#!/bin/bash
# 清理Mosquitto日志文件

set -e

echo "=========================================="
echo "清理Mosquitto日志文件"
echo "=========================================="
echo ""

LOG_DIR="/var/log/mosquitto"
KEEP_FILES=3  # 保留最新的N个日志文件

# 检查目录是否存在
if [ ! -d "$LOG_DIR" ]; then
    echo "错误: 日志目录不存在: $LOG_DIR"
    exit 1
fi

# 显示清理前的状态
echo "清理前的日志文件:"
sudo du -sh "$LOG_DIR"/* 2>/dev/null | sort -hr | head -10
echo ""

TOTAL_SIZE_BEFORE=$(sudo du -sh "$LOG_DIR" 2>/dev/null | cut -f1)
echo "清理前总大小: $TOTAL_SIZE_BEFORE"
echo ""

# 清理旧的压缩日志文件（只保留最新的N个）
echo "清理旧的压缩日志文件（保留最新的 $KEEP_FILES 个）..."
OLD_GZ_FILES=$(sudo find "$LOG_DIR" -name "*.log.*.gz" -type f | wc -l)
if [ "$OLD_GZ_FILES" -gt "$KEEP_FILES" ]; then
    echo "   发现 $OLD_GZ_FILES 个压缩日志文件"
    sudo find "$LOG_DIR" -name "*.log.*.gz" -type f -printf '%T@ %p\n' | \
        sort -rn | tail -n +$((KEEP_FILES + 1)) | cut -d' ' -f2- | \
        while read file; do
            SIZE=$(sudo du -sh "$file" 2>/dev/null | cut -f1)
            echo "   删除: $(basename "$file") ($SIZE)"
            sudo rm -f "$file"
        done
    echo "   ✅ 已清理旧压缩日志文件"
else
    echo "   ✅ 压缩日志文件数量正常（$OLD_GZ_FILES 个）"
fi
echo ""

# 清理当前日志文件（如果太大）
echo "检查当前日志文件大小..."
CURRENT_LOG="$LOG_DIR/mosquitto.log"
if [ -f "$CURRENT_LOG" ]; then
    SIZE=$(sudo du -sh "$CURRENT_LOG" 2>/dev/null | cut -f1 | sed 's/[^0-9.]//g')
    SIZE_MB=$(echo "$SIZE" | awk '{print int($1)}')
    
    # 如果当前日志文件超过100MB，清空它
    if [ "$SIZE_MB" -gt 100 ] 2>/dev/null; then
        echo "   当前日志文件过大（${SIZE}MB），正在清空..."
        sudo truncate -s 0 "$CURRENT_LOG"
        echo "   ✅ 当前日志文件已清空"
    else
        echo "   ✅ 当前日志文件大小正常（${SIZE}MB）"
    fi
fi
echo ""

# 显示清理后的状态
echo "清理后的日志文件:"
sudo du -sh "$LOG_DIR"/* 2>/dev/null | sort -hr | head -10
echo ""

TOTAL_SIZE_AFTER=$(sudo du -sh "$LOG_DIR" 2>/dev/null | cut -f1)
echo "清理后总大小: $TOTAL_SIZE_AFTER"
echo ""

echo "=========================================="
echo "清理完成！"
echo "=========================================="
