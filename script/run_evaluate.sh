#!/bin/bash
# 使用 sudo 运行评估脚本（实际修改 Mosquitto 配置）

# 设置 PYTHONPATH
export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH

# 自动检测并设置 EMQTT_BENCH_PATH（如果未设置）
if [ -z "$EMQTT_BENCH_PATH" ]; then
    # 尝试在常见位置查找 emqtt_bench
    POSSIBLE_PATHS=(
        "/home/qincai/userDir/mosquitto/BrokerPressTest/tool/emqtt_bench_release/bin/emqtt_bench"
        "/home/qincai/userDir/mosquitto/BrokerPressTest/tool/emqtt-bench-source/bin/emqtt_bench"
        "$(which emqtt_bench 2>/dev/null)"
        "$HOME/emqtt-bench/emqtt_bench"
        "$HOME/emqtt-bench/_build/default/bin/emqtt_bench"
    )
    
    for path in "${POSSIBLE_PATHS[@]}"; do
        if [ -n "$path" ] && [ -f "$path" ] && [ -x "$path" ]; then
            export EMQTT_BENCH_PATH="$path"
            echo "自动检测到 emqtt_bench: $EMQTT_BENCH_PATH"
            break
        fi
    done
fi

# 自动检测并设置 MOSQUITTO_PID
# 优先使用环境变量中已设置的 PID，但会验证其有效性
if [ -n "$MOSQUITTO_PID" ]; then
    # 如果环境变量已设置，先验证 PID 是否有效
    # 使用 /proc 目录检查，不依赖 kill 命令（避免权限问题）
    if [ ! -d "/proc/$MOSQUITTO_PID" ]; then
        echo "警告: 环境变量中的 PID $MOSQUITTO_PID 对应的进程不存在，将重新检测..."
        unset MOSQUITTO_PID
    else
        # 验证进程名是否包含 mosquitto
        if ! grep -q "mosquitto" "/proc/$MOSQUITTO_PID/comm" 2>/dev/null; then
            echo "警告: PID $MOSQUITTO_PID 不是 mosquitto 进程，将重新检测..."
            unset MOSQUITTO_PID
        fi
    fi
fi

# 如果未设置或无效，重新检测
if [ -z "$MOSQUITTO_PID" ]; then
    # 尝试获取 Mosquitto 进程 PID
    # 如果有多个进程，选择第一个（通常是主进程）
    MOSQUITTO_PID=$(pgrep -o mosquitto)
    
    if [ -z "$MOSQUITTO_PID" ]; then
        echo "错误: 未找到 Mosquitto 进程，请确保 Mosquitto 正在运行"
        echo ""
        echo "请执行以下步骤："
        echo "1. 启动 Mosquitto: sudo systemctl start mosquitto"
        echo "2. 检查状态: systemctl status mosquitto"
        echo "3. 或手动设置 PID: export MOSQUITTO_PID=\$(pgrep mosquitto)"
        exit 1
    fi
    
    # 如果有多个进程，显示警告
    PID_COUNT=$(pgrep -c mosquitto)
    if [ "$PID_COUNT" -gt 1 ]; then
        echo "警告: 检测到 $PID_COUNT 个 Mosquitto 进程，使用第一个 PID: $MOSQUITTO_PID"
        echo "所有进程: $(pgrep mosquitto | tr '\n' ' ')"
        echo ""
    fi
    
    export MOSQUITTO_PID
fi

# 最终验证 PID 是否有效（进程是否存在）
# 使用 /proc 目录检查，不依赖 kill 命令
if [ ! -d "/proc/$MOSQUITTO_PID" ]; then
    echo "错误: PID $MOSQUITTO_PID 对应的进程不存在"
    echo "请检查 Mosquitto 是否正在运行: systemctl status mosquitto"
    exit 1
fi

# 验证进程名
if ! grep -q "mosquitto" "/proc/$MOSQUITTO_PID/comm" 2>/dev/null; then
    echo "警告: PID $MOSQUITTO_PID 可能不是 mosquitto 进程"
    echo "进程名: $(cat /proc/$MOSQUITTO_PID/comm 2>/dev/null || echo '未知')"
    echo "继续使用此 PID，如果出现问题请手动设置正确的 PID"
fi

echo "使用 PID: $MOSQUITTO_PID"
echo "PYTHONPATH: $PYTHONPATH"
echo "开始评估..."
echo ""

# 使用 sudo -E 保留环境变量运行评估脚本
sudo -E python3 -m tuner.evaluate "$@"
