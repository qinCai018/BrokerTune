#!/bin/bash
# 添加sys_interval配置到Mosquitto主配置文件

CONFIG_FILE="/etc/mosquitto/mosquitto.conf"

echo "=========================================="
echo "添加sys_interval配置到Mosquitto"
echo "=========================================="

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查是否已有sys_interval配置
if grep -qE "^sys_interval" "$CONFIG_FILE"; then
    echo "✅ 配置文件已包含sys_interval配置:"
    grep -E "^sys_interval" "$CONFIG_FILE"
else
    echo "⚠️  配置文件缺少sys_interval配置，正在添加..."
    
    # 备份配置文件
    BACKUP_FILE="${CONFIG_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
    sudo cp "$CONFIG_FILE" "$BACKUP_FILE"
    echo "   备份到: $BACKUP_FILE"
    
    # 检查是否有include_dir配置
    if grep -q "^include_dir" "$CONFIG_FILE"; then
        # 在include_dir之前添加sys_interval
        sudo sed -i '/^include_dir/i sys_interval 10' "$CONFIG_FILE"
    else
        # 在配置文件末尾添加
        echo "" | sudo tee -a "$CONFIG_FILE" > /dev/null
        echo "# Broker Tuner: 添加sys_interval配置（每10秒发布一次$SYS主题）" | sudo tee -a "$CONFIG_FILE" > /dev/null
        echo "sys_interval 10" | sudo tee -a "$CONFIG_FILE" > /dev/null
    fi
    
    echo "✅ 已添加sys_interval配置"
    echo ""
    echo "新配置:"
    grep -E "^sys_interval" "$CONFIG_FILE"
fi

echo ""
echo "=========================================="
echo "重启Mosquitto服务以应用配置..."
sudo systemctl restart mosquitto
sleep 3

echo ""
echo "检查服务状态:"
systemctl status mosquitto --no-pager -l | head -10

echo ""
echo "=========================================="
echo "提示: Broker会在sys_interval时间后发布第一个$SYS消息"
echo "      如果sys_interval=10，最多需要等待10秒"
