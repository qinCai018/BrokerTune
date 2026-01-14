# 强化学习训练启动指南

## 快速启动

### 方法1：使用启动脚本（推荐）

```bash
cd /home/qincai/userDir/BrokerTuner

# 使用默认参数（10万步）
./start_training.sh

# 或指定参数：总步数 保存目录 保存频率
./start_training.sh 100000 ./checkpoints 10000
```

### 方法2：直接使用训练脚本

```bash
cd /home/qincai/userDir/BrokerTuner

./script/run_train.sh \
    --enable-workload \
    --total-timesteps 100000 \
    --save-dir ./checkpoints \
    --save-freq 10000
```

### 方法3：完整参数命令

```bash
cd /home/qincai/userDir/BrokerTuner

./script/run_train.sh \
    --enable-workload \
    --total-timesteps 100000 \
    --save-dir ./checkpoints \
    --save-freq 10000 \
    --device cpu \
    --workload-publishers 100 \
    --workload-subscribers 10 \
    --workload-publisher-interval-ms 15 \
    --workload-message-size 512 \
    --workload-qos 1 \
    --workload-topic "test/topic"
```

## 参数说明

### 必需参数
- `--enable-workload`: **必需**，启用工作负载（训练必须有工作负载）

### 训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-timesteps` | 5,000,000 | 总训练步数 |
| `--save-dir` | `./checkpoints` | 模型保存目录 |
| `--save-freq` | 10,000 | 每隔多少步保存一次checkpoint |
| `--device` | `cpu` | 训练设备（`cpu` 或 `cuda`） |
| `--tau` | 0.00001 | 目标网络软更新系数 |
| `--actor-lr` | 0.00001 | Actor学习率 |
| `--critic-lr` | 0.00001 | Critic学习率 |
| `--gamma` | 0.9 | 折扣因子 |
| `--batch-size` | 16 | 训练批次大小 |

### 工作负载参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--workload-publishers` | 100 | 发布者数量 |
| `--workload-subscribers` | 10 | 订阅者数量 |
| `--workload-topic` | `test/topic` | MQTT主题 |
| `--workload-publisher-interval-ms` | 15 | 每个发布者发布间隔（毫秒） |
| `--workload-message-size` | 512 | 消息大小（字节） |
| `--workload-qos` | 1 | QoS级别（0, 1, 或 2） |
| `--emqtt-bench-path` | 自动检测 | emqtt_bench可执行文件路径 |

## 训练前准备

### 1. 确保Mosquitto正在运行

```bash
# 检查Mosquitto状态
systemctl status mosquitto

# 如果未运行，启动Mosquitto
sudo systemctl start mosquitto
```

### 2. 确保emqtt_bench可用

```bash
# 检查emqtt_bench是否在PATH中
which emqtt_bench

# 或者设置环境变量
export EMQTT_BENCH_PATH=/path/to/emqtt_bench
```

### 3. 测试工作负载（可选）

```bash
# 测试工作负载是否能正常运行
python3 script/test_workload.py --duration 10
```

## 训练示例

### 示例1：快速测试（1000步）

```bash
./script/run_train.sh \
    --enable-workload \
    --total-timesteps 1000 \
    --save-dir ./checkpoints \
    --save-freq 500
```

### 示例2：中等规模训练（10万步）

```bash
./script/run_train.sh \
    --enable-workload \
    --total-timesteps 100000 \
    --save-dir ./checkpoints \
    --save-freq 10000 \
    --workload-publishers 100 \
    --workload-subscribers 10 \
    --workload-publisher-interval-ms 15 \
    --workload-message-size 512 \
    --workload-qos 1
```

### 示例3：大规模训练（100万步）

```bash
./script/run_train.sh \
    --enable-workload \
    --total-timesteps 1000000 \
    --save-dir ./checkpoints \
    --save-freq 10000 \
    --device cpu \
    --workload-publishers 100 \
    --workload-subscribers 10 \
    --workload-publisher-interval-ms 15 \
    --workload-message-size 512 \
    --workload-qos 1
```

## 训练过程监控

### 1. 查看训练日志

```bash
# 查看CSV训练日志
tail -f ./checkpoints/logs/progress.csv

# 查看action和吞吐量日志
tail -f ./checkpoints/action_throughput_log.csv
```

### 2. 使用TensorBoard（如果安装了）

```bash
# 启动TensorBoard
tensorboard --logdir ./checkpoints/logs

# 然后在浏览器中打开 http://localhost:6006
```

### 3. 检查工作负载状态

在另一个终端：

```bash
# 检查工作负载进程
ps aux | grep emqtt_bench | grep -v grep

# 订阅主题查看消息
mosquitto_sub -h 127.0.0.1 -t "test/topic" -v
```

### 4. 检查训练进程

```bash
# 查看训练进程
ps aux | grep "python3 -m tuner.train" | grep -v grep
```

## 训练输出

训练完成后，会在 `--save-dir` 目录生成：

```
checkpoints/
├── ddpg_mosquitto_10000_steps.zip      # Checkpoint文件
├── ddpg_mosquitto_final.zip            # 最终模型 ⭐
├── action_throughput_log.csv           # Action和吞吐量日志 ⭐
├── logs/
│   ├── progress.csv                    # 训练指标日志 ⭐
│   └── events.out.tfevents.*          # TensorBoard日志
└── monitor/
    └── monitor.csv                     # Episode统计
```

## 注意事项

1. **必须使用 `--enable-workload`**：训练必须在有工作负载的情况下进行
2. **需要sudo权限**：修改Broker配置需要root权限，运行时会提示输入密码
3. **训练时间**：训练时间取决于总步数和每步的执行时间
4. **磁盘空间**：确保有足够的磁盘空间保存模型和日志
5. **工作负载稳定性**：确保工作负载稳定运行，系统会自动检测并恢复

## 常见问题

### Q1: 提示"工作负载启动失败"

**解决方法**：
1. 检查emqtt_bench是否安装：`which emqtt_bench`
2. 设置环境变量：`export EMQTT_BENCH_PATH=/path/to/emqtt_bench`
3. 或使用参数：`--emqtt-bench-path /path/to/emqtt_bench`

### Q2: 提示"未找到 Mosquitto 进程"

**解决方法**：
```bash
# 启动Mosquitto
sudo systemctl start mosquitto

# 检查状态
systemctl status mosquitto
```

### Q3: 训练过程中工作负载停止

**说明**：这是正常的，系统会自动检测并重启工作负载（每步检查一次）

### Q4: 需要sudo权限

**说明**：训练脚本会自动使用`sudo -E`运行，需要输入密码。如果配置了无密码sudo，则不需要输入。

## 停止训练

如果需要停止训练：

```bash
# 查找训练进程
ps aux | grep "python3 -m tuner.train" | grep -v grep

# 停止训练进程（替换PID为实际进程ID）
sudo kill -SIGINT <PID>

# 或者使用pkill
sudo pkill -f "python3 -m tuner.train"
```

训练脚本会自动清理工作负载和资源。
