# 吞吐量测试工作流程详解

## 概述

本文档详细说明 `verify/throughput_test.py` 的完整工作流程，包括初始化、测试执行、配置切换、结果保存等各个环节。

## 目录结构

```
verify/
├── throughput_test.py    # 主测试脚本
├── run_test.sh          # 运行脚本
├── README.md            # 使用说明
├── QUICK_START.md       # 快速开始
└── WORKFLOW.md           # 本文档（工作流程详解）
```

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    ThroughputTester                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │  - WorkloadManager (管理工作负载)                 │   │
│  │  - MQTTSampler (采样吞吐量数据)                   │   │
│  │  - BrokerKnobSpace (获取默认配置)                 │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 完整工作流程

### 阶段1：初始化 (ThroughputTester.__init__)

```
1. 设置输出CSV文件路径
   └─ 确保文件保存在verify目录下
   
2. 初始化MQTT配置
   └─ 配置$SYS主题订阅（用于获取吞吐量数据）
      - $SYS/broker/messages/received
      - $SYS/broker/messages/sent
      - $SYS/broker/messages/publish/received
      - $SYS/broker/messages/publish/sent
   
3. 初始化工作负载管理器
   └─ WorkloadManager(broker_host="127.0.0.1", broker_port=1883)
   
4. 初始化配置空间
   └─ BrokerKnobSpace()（用于获取默认配置值）
   
5. 初始化状态变量
   └─ _last_broker_config = None（跟踪上一次的配置）
```

### 阶段2：主测试循环 (run_all_tests)

```
┌─────────────────────────────────────────────────────────┐
│  定义测试配置和用例                                       │
├─────────────────────────────────────────────────────────┤
│  Broker配置:                                             │
│    1. max_inflight_100 (max_inflight_messages=100)      │
│    2. default (所有参数默认)                             │
│                                                          │
│  测试用例: 12种组合                                      │
│    - 消息大小: 256B, 512B, 1024B                         │
│    - QoS: 0, 1                                          │
│    - 发布周期: 10ms, 50ms                                │
│    - 发布端: 100个                                       │
│    - 接收端: 10个                                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  外层循环：遍历Broker配置 (2次)                           │
├─────────────────────────────────────────────────────────┤
│  for broker_config_idx, broker_config in broker_configs:│
│                                                          │
│    ┌────────────────────────────────────────────────┐   │
│    │  配置切换处理（如果不是第一个配置）                │   │
│    ├────────────────────────────────────────────────┤   │
│    │  1. 停止之前的工作负载                            │   │
│    │  2. 等待进程完全终止（5秒）                       │   │
│    │  3. 准备切换到新配置                              │   │
│    └────────────────────────────────────────────────┘   │
│                                                          │
│    ┌────────────────────────────────────────────────┐   │
│    │  内层循环：遍历测试用例 (12次)                    │   │
│    ├────────────────────────────────────────────────┤   │
│    │  for test_case in test_cases:                   │   │
│    │                                                  │   │
│    │    1. 运行单个测试用例 (run_test_case)           │   │
│    │    2. 保存结果到CSV                              │   │
│    │    3. 测试用例之间等待（2秒）                      │   │
│    └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  最终清理                                                 │
├─────────────────────────────────────────────────────────┤
│  1. 确保所有工作负载已停止                                │
│  2. 打印测试摘要                                          │
│  3. 显示结果文件位置                                      │
└─────────────────────────────────────────────────────────┘
```

### 阶段3：单个测试用例执行 (run_test_case)

这是最核心的流程，每个测试用例都按以下步骤执行：

```
┌─────────────────────────────────────────────────────────┐
│  步骤1: 应用Broker配置                                    │
├─────────────────────────────────────────────────────────┤
│  apply_broker_config(broker_config)                     │
│                                                          │
│  ├─ 1.1 检测配置切换                                      │
│  │   └─ 如果 _last_broker_config != config.name         │
│  │      └─ 设置 force_restart = True                    │
│  │                                                       │
│  ├─ 1.2 应用配置                                          │
│  │   ├─ 如果 max_inflight_messages 不为 None:          │
│  │   │   └─ 说明：需要设置自定义配置值                    │
│  │   │   └─ 调用 apply_knobs({max_inflight_messages})   │
│  │   │      └─ 写入配置文件（只设置max_inflight_messages）│
│  │   │      └─ 其他配置项使用默认值（不写入）              │
│  │   │      └─ 重启或重载Broker                          │
│  │   │                                                   │
│  │   └─ 如果 max_inflight_messages 为 None:             │
│  │      └─ 说明：使用系统默认配置（所有参数默认）           │
│  │      └─ 清空配置文件（删除所有自定义配置）              │
│  │      └─ 让Mosquitto使用内置默认值                       │
│  │      └─ 重启或重载Broker                              │
│  │                                                       │
│  │  设计原因：                                            │
│  │  - None 作为标记，表示"使用系统默认"                    │
│  │  - 清空配置文件确保没有残留的自定义配置                 │
│  │  - 这样可以准确测试Mosquitto的默认行为                 │
│  │                                                       │
│  ├─ 1.3 验证Broker状态（如果重启）                        │
│  │   ├─ 检查服务状态 (systemctl is-active)              │
│  │   ├─ 检查端口监听 (netstat -tlnp | grep 1883)        │
│  │   └─ 最多等待20秒                                     │
│  │                                                       │
│  ├─ 1.4 等待Broker稳定                                    │
│  │   ├─ 如果重启: 等待5秒 + 12秒（$SYS主题）              │
│  │   └─ 如果重载: 等待3秒                                │
│  │                                                       │
│  └─ 1.5 更新配置状态                                      │
│      └─ _last_broker_config = config.name                │
│                                                          │
│  返回: (broker_restarted, applied_knobs)                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  步骤2: 停止旧工作负载                                     │
├─────────────────────────────────────────────────────────┤
│  1. 检查工作负载是否运行                                   │
│     └─ workload_manager.is_running()                    │
│                                                          │
│  2. 如果运行，停止工作负载                                 │
│     └─ workload_manager.stop()                           │
│        └─ 发送SIGTERM信号给所有emqtt_bench进程           │
│        └─ 等待进程终止                                    │
│                                                          │
│  3. 如果Broker已重启，额外等待                            │
│     └─ 等待5秒确保Broker完全就绪                          │
│                                                          │
│  4. 等待进程完全终止                                      │
│     └─ 等待3秒                                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  步骤3: 创建工作负载配置                                   │
├─────────────────────────────────────────────────────────┤
│  WorkloadConfig(                                         │
│    num_publishers=100,                                   │
│    num_subscribers=10,                                    │
│    topic="test/throughput",                              │
│    message_size=test_case.message_size,                  │
│    qos=test_case.qos,                                    │
│    publisher_interval_ms=test_case.publisher_interval_ms, │
│    duration=0  # 持续运行                                │
│  )                                                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  步骤4: 启动新的工作负载                                   │
├─────────────────────────────────────────────────────────┤
│  workload_manager.start(config=workload_config)          │
│                                                          │
│  ├─ 4.1 启动订阅者进程                                    │
│  │   └─ emqtt_bench sub -h 127.0.0.1 -p 1883            │
│  │      -c 10 -t test/throughput -q {qos}              │
│  │                                                       │
│  ├─ 4.2 启动发布者进程                                    │
│  │   └─ emqtt_bench pub -h 127.0.0.1 -p 1883            │
│  │      -c 100 -t test/throughput -q {qos}              │
│  │      -I {interval_ms} -s {message_size}               │
│  │                                                       │
│  └─ 4.3 验证工作负载启动                                  │
│      └─ 等待5秒后验证消息发送                             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  步骤5: 等待工作负载稳定                                   │
├─────────────────────────────────────────────────────────┤
│  time.sleep(stable_time_sec)  # 默认30秒                 │
│                                                          │
│  目的: 让工作负载稳定运行，确保吞吐量数据准确              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  步骤6: 统计吞吐量                                         │
├─────────────────────────────────────────────────────────┤
│  MQTTSampler.sample()                                    │
│                                                          │
│  ├─ 6.1 创建MQTT采样器                                    │
│  │   └─ 连接到Broker，订阅$SYS主题                      │
│  │                                                       │
│  ├─ 6.2 第一次采样（5秒）                                 │
│  │   └─ 收集消息计数指标                                 │
│  │      - $SYS/broker/messages/received                 │
│  │      - $SYS/broker/messages/publish/received          │
│  │      - $SYS/broker/messages/sent                      │
│  │                                                       │
│  ├─ 6.3 间隔1秒                                           │
│  │                                                       │
│  ├─ 6.4 第二次采样（5秒）                                 │
│  │   └─ 再次收集消息计数指标                             │
│  │                                                       │
│  └─ 6.5 计算吞吐量                                        │
│      └─ throughput = (metrics2 - metrics1) / 5.0        │
│         （5秒内的平均每秒消息数）                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  步骤7: 停止工作负载                                       │
├─────────────────────────────────────────────────────────┤
│  workload_manager.stop()                                  │
│                                                          │
│  ├─ 发送SIGTERM信号给所有进程                             │
│  ├─ 等待5秒进程终止                                       │
│  └─ 如果未终止，发送SIGKILL强制终止                      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  步骤8: 等待清理完成                                       │
├─────────────────────────────────────────────────────────┤
│  time.sleep(3.0)                                         │
│                                                          │
│  目的: 确保进程完全终止，Broker稳定                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  步骤9: 构建并返回结果                                     │
├─────────────────────────────────────────────────────────┤
│  返回字典包含:                                            │
│  - 基本信息: broker_config, message_size, qos, ...      │
│  - 吞吐量: throughput                                     │
│  - 所有Broker配置项: max_inflight_messages, ...          │
└─────────────────────────────────────────────────────────┘
```

## 关键流程详解

### 0. 配置模式判断：为什么使用 `is not None`？

**核心问题**：如何区分"设置自定义值"和"使用系统默认值"？

**解决方案**：使用 `None` 作为特殊标记

```python
@dataclass
class BrokerConfig:
    name: str
    max_inflight_messages: int | None = None
    # None 表示使用系统默认值，非None表示设置自定义值
```

#### 两种配置模式对比

| 配置模式 | 代码值 | 判断条件 | 配置文件操作 | Broker实际使用的值 |
|---------|--------|---------|-------------|------------------|
| **自定义配置** | `max_inflight_messages = 100` | `is not None` | 写入 `max_inflight_messages 100` | **100** |
| **系统默认配置** | `max_inflight_messages = None` | `is None` | **清空配置文件** | **Mosquitto内置默认值（通常是20）** |

#### 详细说明

**1. 自定义配置模式** (`max_inflight_messages = 100`)

```python
if config.max_inflight_messages is not None:
    # 写入配置文件: max_inflight_messages 100
    knobs = {"max_inflight_messages": 100}
    apply_knobs(knobs)
```

**行为**：
- ✅ 在配置文件中写入 `max_inflight_messages 100`
- ✅ 其他配置项不写入（使用Mosquitto默认值）
- ✅ Broker使用100作为max_inflight_messages的值

**配置文件内容**：
```
# 自动生成：BrokerTuner knobs
max_inflight_messages 100
```

**2. 系统默认配置模式** (`max_inflight_messages = None`)

```python
else:  # max_inflight_messages is None
    # 清空配置文件
    # 让Mosquitto使用系统内置默认值
```

**行为**：
- ✅ **清空配置文件**（删除所有自定义配置）
- ✅ 让Mosquitto使用其**内置的默认值**（不是代码中的默认值）
- ✅ 测试"完全没有自定义配置"的情况

**配置文件内容**：
```
# 默认配置（所有参数使用系统默认值）
```

#### 为什么需要清空配置文件？

**问题场景**：

如果不清空配置文件，从 `max_inflight_100` 切换到 `default` 时：

```
测试1: max_inflight_100
  └─ 配置文件: max_inflight_messages 100
  
测试2: default（不清空）
  └─ 配置文件: max_inflight_messages 100  ← 残留！
  └─ 实际使用的还是100，不是系统默认值20
```

**解决方案**：

```
测试1: max_inflight_100
  └─ 配置文件: max_inflight_messages 100
  
测试2: default（清空文件）
  └─ 配置文件: (清空)
  └─ 实际使用系统默认值20 ✅
```

#### 设计优势

1. **清晰的语义**：
   - `None` = 使用系统默认（特殊标记）
   - 非`None` = 设置自定义值

2. **配置隔离**：
   - 清空文件确保测试纯净
   - 避免配置残留影响测试结果

3. **准确对比**：
   - 可以准确对比"自定义配置" vs "系统默认配置"
   - 测试结果更有参考价值

4. **灵活性**：
   - 可以轻松扩展支持其他配置项
   - 只需在BrokerConfig中添加字段

### 1. 配置切换检测与处理

```python
# 在 apply_broker_config 中
config_changed = (self._last_broker_config is not None and 
                 self._last_broker_config != config.name)

if config_changed:
    # 强制重启Broker
    force_restart = True
    apply_knobs(knobs, force_restart=True)
```

**流程**:
1. 比较当前配置与上一次配置
2. 如果不同，设置 `force_restart=True`
3. 强制重启Broker（而不是reload）
4. 验证Broker重启成功
5. 更新 `_last_broker_config`

### 2. Broker重启验证

```python
# 验证Broker是否正常运行
while waited < max_wait:
    # 检查服务状态
    systemctl is-active mosquitto
    
    # 检查端口监听
    netstat -tlnp | grep 1883
    
    if 服务active and 端口监听:
        break
    time.sleep(1.0)
```

**验证步骤**:
1. 检查systemd服务状态
2. 检查端口1883是否监听
3. 最多等待20秒
4. 每5秒打印一次进度

### 3. 工作负载重启机制

**每个测试用例都会**:
1. 停止旧工作负载（如果存在）
2. 等待进程完全终止
3. 启动新的工作负载
4. 验证工作负载运行

**Broker重启后**:
1. 检测到Broker重启（`broker_restarted=True`）
2. 额外等待5秒确保Broker完全就绪
3. 停止旧工作负载（Broker重启导致连接断开）
4. 启动新的工作负载

### 4. 吞吐量统计方法

```python
# 采样两次，计算差值
metrics1 = sampler.sample(timeout_sec=5.0)  # 第一次采样
time.sleep(1.0)  # 间隔1秒
metrics2 = sampler.sample(timeout_sec=5.0)  # 第二次采样

# 计算吞吐量
messages_diff = metrics2[received_key] - metrics1[received_key]
throughput = messages_diff / 5.0  # 每秒消息数
```

**优先级**:
1. `$SYS/broker/messages/received`（首选）
2. `$SYS/broker/messages/publish/received`（备选）
3. `$SYS/broker/messages/sent`（备用）

## 时间线示例

以单个测试用例为例（假设Broker已重启）：

```
时间轴（秒）    操作
─────────────────────────────────────────────────
0             开始测试用例
0-17          应用Broker配置（重启Broker）
              - 写入配置: 1秒
              - 重启Broker: 2秒
              - 等待稳定: 5秒
              - 验证状态: 最多20秒（实际可能更快）
              - 等待$SYS主题: 12秒
17-20         停止旧工作负载 + 等待
20-25         启动新工作负载 + 验证
25-55         等待工作负载稳定（30秒）
55-66         统计吞吐量
              - 第一次采样: 5秒
              - 间隔: 1秒
              - 第二次采样: 5秒
66-69         停止工作负载 + 等待
─────────────────────────────────────────────────
总计: 约69秒/测试用例
```

**24个测试用例总时间**: 约27-30分钟

## 数据流

```
┌──────────────┐
│  Broker配置   │
│  (knobs)     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ apply_knobs() │
│ 写入配置文件  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 重启/重载     │
│  Mosquitto   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  启动工作负载  │
│  emqtt_bench │
└──────┬───────┘
       │
       │ 发送消息
       ▼
┌──────────────┐
│   Mosquitto   │
│    Broker     │
└──────┬───────┘
       │
       │ 发布$SYS指标
       ▼
┌──────────────┐
│ MQTTSampler  │
│  订阅$SYS    │
└──────┬───────┘
       │
       │ 计算差值
       ▼
┌──────────────┐
│   吞吐量      │
│  (msg/s)     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   CSV文件     │
│  (结果保存)   │
└──────────────┘
```

## 错误处理

### 1. Broker配置应用失败

```python
try:
    apply_knobs(knobs, force_restart=True)
except Exception as e:
    # 记录错误，继续下一个测试
    error_result = {...}
    results.append(error_result)
```

### 2. 工作负载启动失败

```python
try:
    workload_manager.start(config)
except Exception as e:
    # 返回错误结果，包含所有配置项
    return {
        "throughput": 0.0,
        "error": str(e),
        ...
    }
```

### 3. 吞吐量统计失败

```python
try:
    sampler = MQTTSampler(...)
    metrics = sampler.sample(...)
except Exception as e:
    # 设置吞吐量为0，继续执行
    throughput = 0.0
```

### 4. 用户中断 (Ctrl+C)

```python
except KeyboardInterrupt:
    # 保存已完成的测试结果
    tester.save_results()
    tester.print_summary()
```

## 结果保存机制

### CSV文件结构

```csv
broker_config,message_size,qos,publisher_interval_ms,num_publishers,num_subscribers,throughput,error,
max_inflight_messages,max_inflight_bytes,max_queued_messages,max_queued_bytes,queue_qos0_messages,
memory_limit,persistence,autosave_interval,set_tcp_nodelay,max_packet_size,message_size_limit
```

### 保存时机

1. **每完成一个测试用例**：立即保存（增量保存）
2. **测试失败时**：保存错误结果
3. **用户中断时**：保存已完成的结果
4. **所有测试完成**：最终保存

### 文件位置

- 默认路径: `verify/throughput_test_results.csv`
- 可通过 `--output` 参数自定义

## 状态跟踪

### 关键状态变量

```python
self._last_broker_config: str | None
# 跟踪上一次使用的Broker配置名称
# 用于检测配置切换

self.results: List[Dict[str, Any]]
# 存储所有测试结果
# 每完成一个测试就追加
```

### 状态转换

```
初始状态
  └─ _last_broker_config = None
  
第一个配置 (max_inflight_100)
  └─ _last_broker_config = "max_inflight_100"
  
配置切换检测
  └─ _last_broker_config != "default"
  └─ 强制重启Broker
  
第二个配置 (default)
  └─ _last_broker_config = "default"
```

## 依赖关系

```
throughput_test.py
  ├─ environment.knobs
  │   ├─ apply_knobs()          # 应用Broker配置
  │   └─ BrokerKnobSpace        # 获取默认配置
  │
  ├─ environment.utils
  │   └─ MQTTSampler            # 采样吞吐量数据
  │
  ├─ environment.config
  │   └─ MQTTConfig             # MQTT连接配置
  │
  └─ script.workload
      ├─ WorkloadManager        # 管理工作负载
      └─ WorkloadConfig         # 工作负载配置
```

## 关键设计决策

### 1. 为什么使用 `max_inflight_messages is not None` 来判断？

**设计原因**:

这个判断用于区分两种配置模式：

#### 模式1: 自定义配置 (`max_inflight_messages = 100`)
```python
if config.max_inflight_messages is not None:
    # 设置自定义值
    knobs = {"max_inflight_messages": 100}
    apply_knobs(knobs)  # 写入配置文件
```

**行为**:
- 在配置文件中写入 `max_inflight_messages 100`
- 其他配置项不写入（使用Mosquitto默认值）
- 通过 `apply_knobs()` 统一处理

#### 模式2: 系统默认配置 (`max_inflight_messages = None`)
```python
else:  # max_inflight_messages is None
    # 清空配置文件
    # 让Mosquitto使用系统内置默认值
```

**行为**:
- **清空配置文件**（删除所有自定义配置）
- 让Mosquitto使用其**内置的默认值**（不是BrokerKnobSpace的默认值）
- 这样可以测试"完全没有自定义配置"的情况

**为什么这样设计？**

1. **测试对比需求**:
   - 需要对比"自定义配置" vs "系统默认配置"
   - 如果不清空配置文件，可能残留之前的配置

2. **配置隔离**:
   - `None` 作为特殊标记，表示"不使用自定义配置"
   - 清空配置文件确保测试的纯净性

3. **默认值来源不同**:
   - `BrokerKnobSpace.get_default_knobs()` 返回的是代码中定义的默认值
   - Mosquitto系统默认值可能略有不同
   - 清空配置文件可以测试真正的系统默认行为

**示例对比**:

| 配置模式 | max_inflight_messages值 | 配置文件内容 | 实际使用的值 |
|---------|------------------------|-------------|-------------|
| 自定义 | 100 | `max_inflight_messages 100` | 100 |
| 系统默认 | None | (清空文件) | Mosquitto内置默认值（通常是20） |

### 2. 为什么每个测试用例都重启工作负载？

**原因**:
- 确保测试之间相互独立
- 避免上一个测试的影响
- 确保工作负载参数完全匹配当前测试用例

### 3. 为什么配置切换时强制重启Broker？

**原因**:
- 某些配置可能需要完全重启才能生效
- 确保配置完全切换，不留残留
- 保证测试结果的准确性

### 4. 为什么使用两次采样计算吞吐量？

**原因**:
- $SYS主题发布的是累计值，不是瞬时值
- 通过差值计算得到实际的消息增量
- 5秒采样窗口提供稳定的平均值

### 5. 为什么等待时间这么长？

**时间分配**:
- Broker重启后等待17秒：确保服务完全就绪
- 工作负载稳定30秒：确保吞吐量数据准确
- 采样10秒：获取足够的统计数据
- 进程清理3秒：确保完全终止

## 性能优化建议

1. **并行测试**（如果支持）：
   - 可以同时测试多个工作负载组合
   - 需要确保Broker配置一致

2. **减少等待时间**（测试用）：
   - 可以缩短稳定运行时间（如10秒）
   - 但可能影响吞吐量准确性

3. **批量保存**：
   - 当前每完成一个测试就保存
   - 可以改为每N个测试保存一次（但风险是丢失数据）

## 故障排查流程

```
问题: 测试失败
  │
  ├─ 检查Broker状态
  │   └─ systemctl status mosquitto
  │
  ├─ 检查工作负载进程
  │   └─ ps aux | grep emqtt_bench
  │
  ├─ 检查$SYS主题
  │   └─ mosquitto_sub -t '$SYS/#' -v
  │
  └─ 查看错误日志
      └─ CSV文件中的error列
```

## 总结

整个工作流程设计确保了：

1. ✅ **配置隔离**：每个Broker配置独立测试
2. ✅ **工作负载隔离**：每个测试用例使用全新的工作负载
3. ✅ **数据准确性**：充分的等待时间确保数据稳定
4. ✅ **错误恢复**：完善的错误处理机制
5. ✅ **结果完整性**：每行结果包含所有配置项和测试参数
6. ✅ **增量保存**：每完成一个测试就保存，避免数据丢失

整个流程约需27-30分钟完成24个测试用例，生成包含所有配置项和吞吐量数据的CSV文件。
