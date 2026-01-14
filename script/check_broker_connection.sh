#!/bin/bash
# 检查Mosquitto Broker连接状态

echo "=========================================="
echo "Mosquitto Broker 连接检查"
echo "=========================================="

# 1. 检查Mosquitto服务状态
echo ""
echo "1. 检查Mosquitto服务状态:"
systemctl status mosquitto --no-pager -l | head -20

# 2. 检查Mosquitto进程
echo ""
echo "2. 检查Mosquitto进程:"
ps aux | grep mosquitto | grep -v grep

# 3. 检查端口监听
echo ""
echo "3. 检查端口1883是否监听:"
sudo netstat -tlnp | grep 1883 || ss -tlnp | grep 1883 || echo "未找到端口1883监听"

# 4. 尝试连接测试
echo ""
echo "4. 尝试连接测试:"
timeout 2 mosquitto_sub -h 127.0.0.1 -p 1883 -t "test/topic" -C 1 2>&1 || echo "连接失败"

# 5. 检查配置文件
echo ""
echo "5. 检查配置文件:"
if [ -f /etc/mosquitto/mosquitto.conf ]; then
    echo "主配置文件存在: /etc/mosquitto/mosquitto.conf"
    echo "监听端口配置:"
    grep -E "^listener|^port" /etc/mosquitto/mosquitto.conf || echo "未找到监听配置"
else
    echo "主配置文件不存在: /etc/mosquitto/mosquitto.conf"
fi

# 6. 检查日志
echo ""
echo "6. 最近的Mosquitto日志:"
sudo journalctl -u mosquitto -n 20 --no-pager 2>/dev/null || tail -20 /var/log/mosquitto/mosquitto.log 2>/dev/null || echo "无法访问日志"

echo ""
echo "=========================================="
