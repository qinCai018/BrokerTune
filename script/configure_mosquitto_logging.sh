#!/bin/bash
# 配置Mosquitto日志以减少磁盘占用
# 1. 设置日志级别为warning（减少详细日志）
# 2. 配置logrotate限制日志文件大小和数量

set -e

echo "=========================================="
echo "配置Mosquitto日志以减少磁盘占用"
echo "=========================================="
echo ""

CONFIG_FILE="/etc/mosquitto/conf.d/broker_tuner_logging.conf"
LOGROTATE_FILE="/etc/logrotate.d/mosquitto"

# 1. 配置Mosquitto日志级别
echo "1. 配置Mosquitto日志级别..."
if [ ! -f "$CONFIG_FILE" ]; then
    echo "   创建日志配置文件: $CONFIG_FILE"
    sudo tee "$CONFIG_FILE" > /dev/null << 'EOF'
# Broker Tuner: 日志配置（减少磁盘占用）
# 设置日志级别为warning，只记录警告和错误
log_type warning
log_type error

# 可选：禁用文件日志，只使用syslog（如果不需要文件日志）
# log_dest file /var/log/mosquitto/mosquitto.log
# 或者完全禁用日志（不推荐，但可以最大程度减少磁盘占用）
# log_dest none
EOF
    echo "   ✅ 日志配置文件已创建"
else
    echo "   ⚠️  配置文件已存在: $CONFIG_FILE"
    echo "   当前内容:"
    sudo cat "$CONFIG_FILE"
fi
echo ""

# 2. 配置logrotate限制日志文件大小
echo "2. 配置logrotate限制日志文件大小和数量..."
if [ ! -f "$LOGROTATE_FILE" ]; then
    echo "   创建logrotate配置: $LOGROTATE_FILE"
    sudo tee "$LOGROTATE_FILE" > /dev/null << 'EOF'
/var/log/mosquitto/*.log {
    daily
    rotate 3
    compress
    delaycompress
    missingok
    notifempty
    create 0640 mosquitto mosquitto
    sharedscripts
    postrotate
        systemctl reload mosquitto > /dev/null 2>&1 || true
    endscript
}

# 限制压缩日志文件数量（只保留最近3个）
/var/log/mosquitto/*.log.*.gz {
    daily
    rotate 3
    missingok
    notifempty
}
EOF
    echo "   ✅ logrotate配置已创建"
    echo "   配置说明:"
    echo "     - 每天轮转日志"
    echo "     - 只保留最近3个日志文件"
    echo "     - 自动压缩旧日志"
else
    echo "   ⚠️  logrotate配置已存在: $LOGROTATE_FILE"
    echo "   当前内容:"
    sudo cat "$LOGROTATE_FILE"
fi
echo ""

# 3. 立即清理旧的日志文件
echo "3. 清理旧的日志文件..."
OLD_LOGS=$(sudo find /var/log/mosquitto -name "*.log.*.gz" -type f | wc -l)
if [ "$OLD_LOGS" -gt 3 ]; then
    echo "   发现 $OLD_LOGS 个旧日志文件，只保留最新的3个..."
    sudo find /var/log/mosquitto -name "*.log.*.gz" -type f -printf '%T@ %p\n' | \
        sort -rn | tail -n +4 | cut -d' ' -f2- | \
        sudo xargs rm -f 2>/dev/null || true
    echo "   ✅ 旧日志文件已清理"
else
    echo "   ✅ 日志文件数量正常（$OLD_LOGS 个）"
fi
echo ""

# 4. 检查当前日志大小
echo "4. 当前日志文件大小:"
sudo du -sh /var/log/mosquitto/* 2>/dev/null | sort -hr | head -5
echo ""

# 5. 重新加载Mosquitto配置
echo "5. 重新加载Mosquitto配置..."
if systemctl is-active --quiet mosquitto; then
    sudo systemctl reload mosquitto 2>/dev/null || {
        echo "   ⚠️  重新加载失败，尝试重启..."
        sudo systemctl restart mosquitto
    }
    echo "   ✅ Mosquitto配置已重新加载"
else
    echo "   ⚠️  Mosquitto服务未运行，配置将在下次启动时生效"
fi
echo ""

echo "=========================================="
echo "配置完成！"
echo "=========================================="
echo ""
echo "日志配置说明:"
echo "  - 日志级别: warning（只记录警告和错误）"
echo "  - 日志轮转: 每天，只保留最近3个文件"
echo "  - 自动压缩: 是"
echo ""
echo "如需进一步减少日志，可以:"
echo "  1. 完全禁用文件日志（编辑 $CONFIG_FILE，添加 log_dest none）"
echo "  2. 手动清理旧日志: sudo ./script/cleanup_mosquitto_logs.sh"
echo ""
