# 吞吐量为0问题修复指南

## 问题描述

训练过程中，`action_throughput_log.csv` 中所有步骤的 `throughput` 都是 `0.0`。

## 问题分析

### 吞吐量的计算流程

```
1. MQTTSampler.sample() 
   ↓
2. 订阅 $SYS/# 主题，等待 timeout_sec 秒
   ↓
3. 收集消息到 broker_metrics 字典
   ↓
4. build_state_vector() 提取 messages_received
   ↓
5. msg_rate_norm = messages_received / 10000.0
   ↓
6. throughput = state[1] = msg_rate_norm
```

### 可能的原因

1. **Broker未配置sys_interval** ⚠️
   - Broker默认不发布 `$SYS` 主题
   - 需要配置 `sys_interval 10`（每10秒发布一次）

2. **采样时间太短** ⚠️
   - 默认 `timeout_sec = 2.0` 秒
   - 如果 `sys_interval = 10`，可能2秒内收不到消息
   - **已修复**：增加到5秒

3. **Broker重启后MQTT连接断开** ⚠️
   - Broker重启后，MQTT连接会断开
   - 需要重新连接才能收到消息
   - **已修复**：在reset和_sample_state中检查并重新连接

4. **工作负载未运行** ⚠️
   - 如果没有工作负载，Broker不会收到消息
   - `$SYS/broker/messages/received` 会保持为0

5. **Broker刚重启，$SYS主题还未发布** ⚠️
   - Broker重启后，需要等待 `sys_interval` 时间才会发布第一个$SYS消息
   - 如果 `sys_interval = 10`，最多需要等待10秒

## 已实施的修复

### 1. ✅ 增加采样超时时间

**位置**：`environment/config.py`

```python
timeout_sec: float = 5.0  # 从2.0秒增加到5.0秒
```

**原因**：确保有足够时间收到$SYS主题消息

### 2. ✅ 改进MQTT采样器连接管理

**位置**：`environment/broker.py`

**改进**：
- 延迟初始化采样器（在reset时创建）
- 在`_sample_state()`中检查连接状态
- 如果连接断开，自动重新创建采样器
- 添加连接状态检查和重连逻辑

### 3. ✅ 添加诊断信息

**位置**：`environment/broker.py` 的 `_sample_state()`

**改进**：
- 如果未收到任何指标，打印警告
- 提示可能的原因和解决方案

## 诊断步骤

### 步骤1：运行诊断脚本

```bash
cd /home/qincai/userDir/BrokerTuner
python3 script/diagnose_throughput_issue.py
```

这个脚本会检查：
1. Broker状态
2. sys_interval配置
3. $SYS主题是否可用
4. 工作负载是否运行

### 步骤2：检查Broker sys_interval配置

```bash
# 检查主配置文件
grep -E "sys_interval" /etc/mosquitto/mosquitto.conf

# 如果没有，添加配置
sudo bash -c 'echo "sys_interval 10" >> /etc/mosquitto/mosquitto.conf'

# 重启Broker
sudo systemctl restart mosquitto
```

### 步骤3：测试$SYS主题

```bash
# 订阅$SYS主题，等待消息
mosquitto_sub -h 127.0.0.1 -p 1883 -t "\$SYS/#" -v -W 15

# 应该能看到类似输出：
# $SYS/broker/clients/connected 110
# $SYS/broker/messages/received 12345
# ...
```

### 步骤4：检查工作负载

```bash
# 检查工作负载进程
ps aux | grep emqtt_bench | grep -v grep

# 应该能看到发布者和订阅者进程
```

## 修复方案

### 方案1：确保Broker配置sys_interval（必需）⭐

```bash
# 检查配置
grep -E "sys_interval" /etc/mosquitto/mosquitto.conf

# 如果没有，添加配置
sudo bash -c 'echo "sys_interval 10" >> /etc/mosquitto/mosquitto.conf'

# 重启Broker
sudo systemctl restart mosquitto

# 等待sys_interval时间（10秒）后测试
sleep 10
mosquitto_sub -h 127.0.0.1 -p 1883 -t "\$SYS/broker/messages/received" -C 1
```

### 方案2：使用修复脚本

已创建修复脚本 `script/add_sys_interval.sh`（如果存在）：

```bash
sudo ./script/add_sys_interval.sh
```

### 方案3：验证修复

```bash
# 1. 运行诊断脚本
python3 script/diagnose_throughput_issue.py

# 2. 手动测试$SYS主题
mosquitto_sub -h 127.0.0.1 -p 1883 -t "\$SYS/broker/messages/received" -C 1

# 3. 如果收到消息，重新运行训练
./script/run_train.sh --enable-workload --total-timesteps 1000
```

## 代码改进说明

### 1. 采样超时时间增加

**文件**：`environment/config.py`

**修改**：
```python
timeout_sec: float = 5.0  # 从2.0秒增加到5.0秒
```

**原因**：
- 如果 `sys_interval = 10`，Broker每10秒发布一次$SYS消息
- 2秒可能不够，增加到5秒提高成功率

### 2. MQTT采样器连接管理

**文件**：`environment/broker.py`

**修改**：
- 延迟初始化采样器（在reset时创建）
- 在`_sample_state()`中检查连接状态
- 如果连接断开，自动重新创建

**原因**：
- Broker重启后，MQTT连接会断开
- 需要重新连接才能收到消息

### 3. 诊断信息

**文件**：`environment/broker.py`

**修改**：
- 如果未收到任何指标，打印警告
- 提示可能的原因

## 验证修复

### 1. 检查配置

```bash
# 检查sys_interval配置
grep -E "sys_interval" /etc/mosquitto/mosquitto.conf

# 应该看到: sys_interval 10
```

### 2. 测试$SYS主题

```bash
# 订阅$SYS主题
mosquitto_sub -h 127.0.0.1 -p 1883 -t "\$SYS/#" -v -W 15

# 应该能看到消息（等待最多15秒）
```

### 3. 运行诊断脚本

```bash
python3 script/diagnose_throughput_issue.py
```

### 4. 重新运行训练

```bash
./script/run_train.sh \
    --enable-workload \
    --total-timesteps 1000 \
    --save-dir ./checkpoints \
    --save-freq 500
```

### 5. 检查吞吐量

```bash
# 查看CSV文件
tail -20 ./checkpoints/action_throughput_log.csv

# 应该能看到非零的throughput值
```

## 常见问题

### Q1: 为什么吞吐量还是0？

**A**: 可能的原因：
1. Broker未配置sys_interval
2. 工作负载未运行
3. Broker刚重启，还未发布$SYS消息

**解决**：
1. 检查sys_interval配置
2. 检查工作负载进程
3. 等待更长时间（sys_interval时间）

### Q2: sys_interval应该设置多少？

**A**: 
- 推荐：`sys_interval 10`（每10秒发布一次）
- 最小值：`sys_interval 1`（每秒发布一次，但会增加Broker负载）
- 训练时建议：`sys_interval 10`（平衡性能和实时性）

### Q3: 采样时间5秒够吗？

**A**: 
- 如果 `sys_interval = 10`，5秒可能不够（最多需要等待10秒）
- 建议：确保 `timeout_sec >= sys_interval`
- 或者设置更小的 `sys_interval`（如5秒）

### Q4: 工作负载运行但吞吐量仍为0？

**A**: 
- 检查Broker是否配置了sys_interval
- 检查$SYS主题是否可用：`mosquitto_sub -h 127.0.0.1 -t "\$SYS/#" -v`
- 检查工作负载是否真的在发送消息

## 总结

**吞吐量为0的主要原因**：
1. ⚠️ Broker未配置sys_interval（最常见）
2. ⚠️ 采样时间太短（已修复：2秒→5秒）
3. ⚠️ Broker重启后MQTT连接断开（已修复：自动重连）
4. ⚠️ 工作负载未运行

**修复步骤**：
1. ✅ 确保Broker配置了 `sys_interval 10`
2. ✅ 确保工作负载正在运行
3. ✅ 重新运行训练（代码已改进）

**验证**：
- 运行诊断脚本：`python3 script/diagnose_throughput_issue.py`
- 检查CSV文件：`tail -20 ./checkpoints/action_throughput_log.csv`
