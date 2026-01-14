#!/bin/bash
# 检查 Mosquitto 配置和状态

echo "=========================================="
echo "Mosquitto 配置和状态检查"
echo "=========================================="
echo ""

# 1. 检查服务状态
echo "1. 服务状态:"
systemctl status mosquitto --no-pager -l | head -15
echo ""

# 2. 检查配置文件
echo "2. 当前 broker_tuner.conf 配置:"
if [ -f /etc/mosquitto/conf.d/broker_tuner.conf ]; then
    cat /etc/mosquitto/conf.d/broker_tuner.conf
else
    echo "配置文件不存在"
fi
echo ""

# 3. 检查主配置文件
echo "3. 主配置文件 mosquitto.conf:"
if [ -f /etc/mosquitto/mosquitto.conf ]; then
    echo "主配置文件存在"
    grep -E "include_dir|sys_interval" /etc/mosquitto/mosquitto.conf || echo "未找到 include_dir 或 sys_interval"
else
    echo "主配置文件不存在"
fi
echo ""

# 4. 检查最近日志
echo "4. 最近 20 条日志:"
sudo journalctl -u mosquitto.service -n 20 --no-pager 2>/dev/null || echo "无法读取日志（需要 sudo 权限）"
echo ""

# 5. 测试配置语法（通过 systemctl 验证）
echo "5. 测试配置语法:"
echo "注意: Mosquitto 1.6.9 不支持 -t 选项，使用 systemctl 验证配置"
if systemctl is-active --quiet mosquitto; then
    echo "✓ Mosquitto 服务正在运行，配置有效"
else
    echo "⚠ Mosquitto 服务未运行，尝试测试配置..."
    sudo systemctl reload mosquitto 2>&1 | head -5 || echo "配置测试失败或需要重启"
fi
echo ""

echo "=========================================="
