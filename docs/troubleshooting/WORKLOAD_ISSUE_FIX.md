# 工作负载问题修复说明

## 问题描述

用户发现训练过程中工作负载可能没有成功运行，因为在另一个终端使用 `mosquitto_sub -t "test/topic"` 没有收到消息。

## 问题分析

### 1. 工作负载主题配置
- **主题名称**：`test/topic`（默认值）
- **配置位置**：`tuner/train.py` 第123行，默认值为 `"test/topic"`
- **验证**：主题配置正确

### 2. 可能的问题原因

1. **Broker 重启导致连接断开**
   - 当 `apply_knobs()` 调用 `systemctl restart mosquitto` 时，所有 MQTT 连接都会断开
   - `emqtt_bench` 进程可能因为连接断开而退出
   - 工作负载进程退出后，不会再发送消息

2. **工作负载进程检查不够频繁**
   - 原来的健康检查每 1000 步检查一次
   - 如果工作负载在步数 100-999 之间退出，可能要等很久才会被发现

3. **缺少自动重启机制**
   - 工作负载健康检查只警告，不重启
   - 没有在 Broker 重启后自动重启工作负载

## 修复方案

### 1. ✅ 添加工作负载重启功能
- 在 `WorkloadManager` 中添加 `restart()` 方法
- 保存最后一次使用的配置，用于重启
- 允许 `start()` 方法在已运行时先停止再启动

### 2. ✅ 改进工作负载健康检查
- 将检查频率从每 1000 步改为每 100 步
- 当检测到工作负载停止时，自动尝试重启
- 记录重启次数，便于监控

### 3. ✅ 工作负载主题确认
- 主题：`test/topic`（默认）
- 可以通过 `--workload-topic` 参数自定义

## 修复的代码

### `script/workload.py`
1. 添加 `_last_config` 属性保存配置
2. 修改 `start()` 方法，允许重启
3. 添加 `restart()` 方法

### `tuner/train.py`
1. 改进 `WorkloadHealthCheckCallback`，添加自动重启功能
2. 将检查频率从 1000 步改为 100 步

## 验证方法

### 1. 测试工作负载独立运行
```bash
cd /home/qincai/userDir/BrokerTuner
python3 script/test_workload.py --duration 10
```

在另一个终端：
```bash
mosquitto_sub -h 127.0.0.1 -t "test/topic" -v
```

应该能看到消息。

### 2. 检查工作负载进程
```bash
ps aux | grep emqtt_bench | grep -v grep
```

应该能看到发布者和订阅者进程。

### 3. 监控训练过程中的工作负载
训练过程中，如果工作负载停止，会看到：
```
[工作负载健康检查] 工作负载在步数 XXX 时停止运行
[工作负载健康检查] 尝试重启工作负载（第 X 次）...
[工作负载健康检查] 工作负载重启成功
```

## 使用说明

### 工作负载主题
- **默认主题**：`test/topic`
- **自定义主题**：使用 `--workload-topic` 参数

```bash
./script/run_train.sh \
    --enable-workload \
    --workload-topic "custom/topic" \
    ...
```

### 监听消息
在另一个终端监听消息：
```bash
# 监听默认主题
mosquitto_sub -h 127.0.0.1 -t "test/topic" -v

# 监听自定义主题
mosquitto_sub -h 127.0.0.1 -t "custom/topic" -v
```

### 检查工作负载状态
```bash
# 检查进程
ps aux | grep emqtt_bench

# 检查连接数（应该看到工作负载的连接）
mosquitto_sub -h 127.0.0.1 -t '$SYS/broker/clients/connected' -C 1
```

## 注意事项

1. **Broker 重启会断开连接**
   - 这是正常行为
   - 工作负载会自动重启（每 100 步检查一次）

2. **工作负载重启需要时间**
   - 重启过程可能需要几秒钟
   - 在此期间可能暂时没有消息

3. **消息速率**
   - 默认配置：100 个发布者，每 15ms 发布一次
   - 总消息速率：约 6666 msg/s
   - 如果看不到消息，检查 Broker 是否正常运行

4. **QoS 级别**
   - 默认 QoS=1（至少一次）
   - 确保订阅者也使用相同的 QoS 级别

## 故障排查

### 问题1：没有收到消息
1. 检查 Broker 是否运行：`systemctl status mosquitto`
2. 检查工作负载进程：`ps aux | grep emqtt_bench`
3. 检查主题是否正确：确认 `--workload-topic` 参数
4. 检查连接数：`mosquitto_sub -h 127.0.0.1 -t '$SYS/broker/clients/connected' -C 1`

### 问题2：工作负载频繁重启
- 可能是 Broker 频繁重启
- 检查 Broker 日志：`sudo journalctl -u mosquitto.service -n 50`

### 问题3：工作负载启动失败
- 检查 emqtt_bench 是否可用：`which emqtt_bench`
- 检查 Broker 是否可访问：`mosquitto_pub -h 127.0.0.1 -t test -m "test"`

## 总结

修复后，工作负载应该能够：
1. ✅ 在 Broker 重启后自动重启
2. ✅ 每 100 步检查一次健康状态
3. ✅ 自动恢复消息发送
4. ✅ 使用正确的主题 `test/topic`

如果仍然没有收到消息，请检查：
- Broker 是否正常运行
- 工作负载进程是否在运行
- 主题名称是否正确
