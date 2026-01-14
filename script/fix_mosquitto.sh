#!/bin/bash
# 修复 Mosquitto 配置问题

echo "=========================================="
echo "Mosquitto 配置修复脚本"
echo "=========================================="
echo ""

# 1. 备份当前配置
CONFIG_FILE="/etc/mosquitto/conf.d/broker_tuner.conf"
BACKUP_FILE="/etc/mosquitto/conf.d/broker_tuner.conf.backup.$(date +%Y%m%d_%H%M%S)"

if [ -f "$CONFIG_FILE" ]; then
    echo "1. 备份当前配置..."
    sudo cp "$CONFIG_FILE" "$BACKUP_FILE"
    echo "   备份到: $BACKUP_FILE"
    echo ""
    
    # 2. 修复配置：移除 max_packet_size 0 和 message_size_limit 0
    echo "2. 修复配置文件（移除可能导致问题的配置项）..."
    sudo sed -i '/^max_packet_size 0$/d' "$CONFIG_FILE"
    sudo sed -i '/^message_size_limit 0$/d' "$CONFIG_FILE"
    echo "   已移除 max_packet_size 0 和 message_size_limit 0"
    echo ""
    
    # 显示修复后的配置
    echo "3. 修复后的配置内容:"
    cat "$CONFIG_FILE"
    echo ""
fi

# 4. 尝试启动 Mosquitto
echo "4. 尝试启动 Mosquitto..."
if sudo systemctl start mosquitto; then
    echo "   ✅ Mosquitto 启动成功！"
    sleep 2
    if systemctl is-active --quiet mosquitto; then
        echo "   ✅ Mosquitto 服务运行正常"
        echo ""
        echo "当前 PID: $(pgrep -o mosquitto)"
    else
        echo "   ⚠️  Mosquitto 启动后立即退出"
    fi
else
    echo "   ❌ Mosquitto 启动失败"
    echo ""
    echo "请检查日志:"
    echo "  sudo journalctl -u mosquitto.service -n 20 --no-pager"
    echo ""
    echo "或者尝试删除配置文件后重启:"
    echo "  sudo rm $CONFIG_FILE"
    echo "  sudo systemctl start mosquitto"
fi

echo ""
echo "=========================================="
