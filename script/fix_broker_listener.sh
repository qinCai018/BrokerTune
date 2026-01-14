#!/bin/bash
# 修复Mosquitto Broker监听配置

echo "=========================================="
echo "修复Mosquitto Broker监听配置"
echo "=========================================="

CONFIG_FILE="/etc/mosquitto/mosquitto.conf"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查是否已有listener配置
if grep -qE "^listener|^port" "$CONFIG_FILE"; then
    echo "配置文件已包含监听配置:"
    grep -E "^listener|^port" "$CONFIG_FILE"
    echo ""
    echo "如果Broker仍然无法连接，请检查配置是否正确"
else
    echo "配置文件缺少监听配置，正在添加..."
    
    # 备份配置文件
    sudo cp "$CONFIG_FILE" "${CONFIG_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
    
    # 检查是否有include_dir配置
    if grep -q "^include_dir" "$CONFIG_FILE"; then
        echo "检测到include_dir配置，将在主配置文件中添加listener"
        # 在include_dir之前添加listener配置
        sudo sed -i '/^include_dir/i listener 1883\nallow_anonymous true' "$CONFIG_FILE"
    else
        echo "在配置文件末尾添加监听配置"
        echo "" | sudo tee -a "$CONFIG_FILE" > /dev/null
        echo "# Broker Tuner: 添加监听配置" | sudo tee -a "$CONFIG_FILE" > /dev/null
        echo "listener 1883" | sudo tee -a "$CONFIG_FILE" > /dev/null
        echo "allow_anonymous true" | sudo tee -a "$CONFIG_FILE" > /dev/null
    fi
    
    echo "✅ 已添加监听配置"
    echo ""
    echo "新配置:"
    grep -E "^listener|^allow_anonymous" "$CONFIG_FILE" | tail -2
fi

echo ""
echo "=========================================="
echo "重启Mosquitto服务..."
sudo systemctl restart mosquitto
sleep 2

echo ""
echo "检查服务状态:"
systemctl status mosquitto --no-pager -l | head -10

echo ""
echo "检查端口监听:"
sudo netstat -tlnp | grep 1883 || ss -tlnp | grep 1883 || echo "端口仍未监听，请检查日志"

echo ""
echo "=========================================="
