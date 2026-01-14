#!/bin/bash
# 使用 sudo 运行测试脚本（实际写入配置文件）

# 设置环境变量
export MOSQUITTO_PID=$(pgrep mosquitto)

# 如果需要实际写入配置，不设置 BROKER_TUNER_DRY_RUN
# 如果只是测试观察，可以取消下面这行的注释
# export BROKER_TUNER_DRY_RUN=true

# 使用 sudo -E 保留环境变量运行
sudo -E python3 "$(dirname "$0")/test_mosquitto.py" "$@"
