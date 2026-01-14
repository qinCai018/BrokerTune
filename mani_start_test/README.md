# BrokerTuner - 手动启动测试工具

本目录包含用于手动启动 Broker 和收集性能指标的工具。

## 目录结构

```
mani_start_test/
├── start_broker/          # Broker 启动工具
│   ├── copy_config.sh     # 拷贝 Mosquitto 配置文件（可选）
│   ├── start_broker.sh    # 启动 Broker 脚本
│   ├── broker_tuner.conf  # 完整的独立 Mosquitto 配置文件（推荐使用）
│   └── mosquitto.conf     # Mosquitto 配置文件（从系统拷贝，可选）
└── statistical_data/      # 性能指标收集工具
    ├── collect_metrics.py # 性能指标收集脚本（Python）
    ├── run_collect.sh     # 性能指标收集脚本（Bash包装器）
    └── metrics_*.json     # 收集的性能指标数据（运行后生成）
```

## 使用说明

### 1. start_broker - 启动 Broker

#### 1.1 配置文件说明

`broker_tuner.conf` 是一个**完整的独立 Mosquitto 配置文件**，可以直接用于启动 Broker：

```bash
mosquitto -c broker_tuner.conf
```

此配置文件包含所有必要的配置项：
- 监听端口配置（listener 1883）
- 匿名连接允许（allow_anonymous true）
- $SYS 主题配置（sys_interval 10）
- 消息队列配置（max_inflight_messages 等）
- 其他必要的配置项

**注意**: `broker_tuner.conf` 是一个完整的独立配置文件，不是 `/etc/mosquitto/conf.d/` 下的片段配置文件。

#### 1.2 启动 Broker

直接使用启动脚本：

```bash
cd mani_start_test/start_broker
./start_broker.sh
```

脚本会：
1. 优先使用 `broker_tuner.conf`（完整的独立配置文件）
2. 如果不存在，则使用 `mosquitto.conf`（从系统拷贝的配置文件）
3. 停止现有的 Mosquitto 服务
4. 等待端口 1883 释放
5. 使用本地配置文件启动 Mosquitto Broker（前台运行）

**注意**: 脚本会在前台运行 Broker，按 `Ctrl+C` 停止。

#### 1.3 可选：从系统拷贝配置文件

如果需要参考系统配置文件，可以使用：

```bash
./copy_config.sh
```

这将从 `/etc/mosquitto/mosquitto.conf` 拷贝配置文件到当前目录的 `mosquitto.conf`。但通常不需要这样做，因为 `broker_tuner.conf` 已经是完整的配置文件。

### 2. statistical_data - 性能指标收集

#### 2.1 使用 Bash 脚本（推荐）

```bash
cd mani_start_test/statistical_data
./run_collect.sh
```

#### 2.2 使用 Python 脚本

```bash
cd mani_start_test/statistical_data
python3 collect_metrics.py
```

#### 2.3 自定义参数

```bash
# 指定 Broker 地址和端口
python3 collect_metrics.py --broker-host 127.0.0.1 --broker-port 1883

# 指定 emqtt_bench 路径
python3 collect_metrics.py --emqtt-bench-path /path/to/emqtt_bench

# 指定输出文件
python3 collect_metrics.py --output my_metrics.json
```

#### 2.4 工作负载配置

脚本会自动启动以下工作负载：
- **发布者数量**: 100
- **订阅者数量**: 10
- **消息大小**: 1024 字节
- **QoS**: 1
- **发布间隔**: 每个发布者每 10 毫秒发布一条消息

#### 2.5 收集流程

1. 启动工作负载（emqtt_bench）
2. 等待负载稳定（50秒）
3. 收集 Broker 性能指标（通过 MQTT $SYS 主题）
4. 收集进程性能指标（CPU、内存、上下文切换）
5. 保存结果到 JSON 文件

#### 2.6 输出数据格式

收集的性能指标会保存为 JSON 格式，包含：

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "broker_host": "127.0.0.1",
  "broker_port": 1883,
  "workload_config": {
    "num_publishers": 100,
    "num_subscribers": 10,
    "message_size": 1024,
    "qos": 1,
    "publisher_interval_ms": 10
  },
  "broker_metrics": {
    "$SYS/broker/clients/connected": 110,
    "$SYS/broker/load/messages/received/1min": 10000.0,
    ...
  },
  "process_metrics": {
    "pid": 12345,
    "cpu_ratio": 0.25,
    "mem_ratio": 0.15,
    "ctxt_ratio": 0.001
  }
}
```

## 前置要求

### 必需软件

1. **Mosquitto Broker**: 已安装并配置
2. **Python 3**: 用于运行性能指标收集脚本
3. **emqtt_bench**: MQTT 性能测试工具
   ```bash
   git clone https://github.com/emqx/emqtt-bench.git
   cd emqtt-bench
   make
   ```

### Python 依赖

```bash
pip install paho-mqtt numpy
```

## 常见问题

### Q: Broker 启动失败

**A**: 检查：
1. Mosquitto 是否已安装：`which mosquitto`
2. 配置文件是否存在：`ls -l start_broker/mosquitto.conf`
3. 端口 1883 是否被占用：`netstat -tln | grep 1883`

### Q: 性能指标收集失败

**A**: 检查：
1. Broker 是否正在运行：`systemctl status mosquitto` 或 `ps aux | grep mosquitto`
2. Broker 是否配置了 `sys_interval`：`grep sys_interval /etc/mosquitto/mosquitto.conf`
3. emqtt_bench 是否可用：`which emqtt_bench` 或设置 `EMQTT_BENCH_PATH` 环境变量

### Q: 无法连接到 Broker

**A**: 确保：
1. Broker 正在运行
2. Broker 配置了 `listener 1883` 和 `allow_anonymous true`
3. 防火墙允许端口 1883

## 示例工作流

### 完整测试流程

```bash
# 1. 拷贝配置文件
cd mani_start_test/start_broker
./copy_config.sh

# 2. 启动 Broker（在一个终端）
./start_broker.sh

# 3. 在另一个终端收集性能指标
cd ../statistical_data
./run_collect.sh

# 4. 查看结果
cat metrics_*.json | jq .
```

## 注意事项

1. **Broker 配置**: 确保 Broker 配置了 `sys_interval 10` 以启用 $SYS 主题
2. **权限**: 某些操作可能需要 sudo 权限（如拷贝配置文件）
3. **资源占用**: 工作负载会占用一定的 CPU 和网络资源
4. **端口冲突**: 确保端口 1883 未被其他程序占用
