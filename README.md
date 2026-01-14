# BrokerTuner

基于强化学习的 Mosquitto Broker 配置参数自动调优系统。针对动态负载场景下静态配置参数难以自动调优的问题，使用 DDPG（Deep Deterministic Policy Gradient）算法在满足时延约束的前提下实现 Broker 吞吐性能的自适应优化。

## 项目简介

BrokerTuner 是一个用于 Mosquitto MQTT Broker 的智能配置调优系统，通过强化学习自动调整以下 11 个关键配置参数：

### QoS 相关配置
- `max_inflight_messages`: 每个客户端同时处于飞行状态的 QoS 1 和 QoS 2 消息数量上限
- `max_inflight_bytes`: 每个客户端同时处于飞行状态的 QoS 1 和 QoS 2 消息总字节数
- `max_queued_messages`: 每个客户端队列中 QoS 1 和 QoS 2 消息数量上限
- `max_queued_bytes`: 限制每个客户端队列中 QoS 1 和 QoS 2 消息的总字节数
- `queue_qos0_messages`: 持久化客户端断开连接期间，其 QoS 0 消息是否加入队列缓存

### 内存相关配置
- `memory_limit`: 为 Broker 内存使用设置硬性上限
- `persistence`: 是否开启持久化
- `autosave_interval`: 持久化启动后，将运行数据写入磁盘的间隔

### 网络延迟相关配置
- `set_tcp_nodelay`: 关闭客户端套接字的 Nagle 算法，降低单条消息的传输延迟

### 通信和协议层配置
- `max_packet_size`: 限制 Broker 能接收的最大 MQTT 报文长度
- `message_size_limit`: Broker 允许接收的最大消息负载

## 环境要求

### 系统要求
- Linux 系统（需要访问 `/proc/[pid]/stat` 和 `/proc/[pid]/status`）
- Python 3.8+
- Mosquitto Broker 已安装并运行
- 具有修改 Mosquitto 配置文件和重启服务的权限（通常需要 root 或 sudo）

### Python 依赖

安装核心依赖：

```bash
pip install stable-baselines3[extra]
pip install gym
pip install paho-mqtt
pip install numpy
pip install torch
```

或者使用 requirements.txt（需要补充上述依赖）：

```bash
pip install -r requirements.txt
pip install stable-baselines3[extra] gym paho-mqtt
```

## 安装步骤

1. **克隆或下载项目**

```bash
cd /home/qincai/userDir/BrokerTuner
```

2. **安装 Python 依赖**

```bash
pip install stable-baselines3[extra] gym paho-mqtt numpy torch
```

3. **配置环境变量**

设置 Mosquitto Broker 的进程 PID：

```bash
export MOSQUITTO_PID=$(pgrep mosquitto)
# 或者手动指定
export MOSQUITTO_PID=12345
```

（可选）自定义配置文件路径：

```bash
export MOSQUITTO_TUNER_CONFIG=/etc/mosquitto/conf.d/broker_tuner.conf
```

4. **确保 Mosquitto Broker 正在运行**

```bash
# 检查 Mosquitto 是否运行
systemctl status mosquitto
# 或
ps aux | grep mosquitto
```

5. **确保 Mosquitto 配置目录存在**

```bash
sudo mkdir -p /etc/mosquitto/conf.d
```

## 项目结构

```
BrokerTuner/
├── environment/          # 环境相关代码
│   ├── __init__.py      # 导出主要类和配置
│   ├── broker.py        # MosquittoBrokerEnv Gym 环境实现
│   ├── config.py         # 环境配置（MQTT、进程采样等）
│   ├── knobs.py          # Broker 配置参数定义和映射
│   └── utils.py          # MQTT 采样、进程指标读取等工具函数
├── model/                # DDPG 模型实现
│   ├── __init__.py      # 导出模型类
│   └── ddpg.py          # CustomActor、CustomCritic、CustomDDPGPolicy
├── tuner/                # 训练和评估脚本
│   ├── __init__.py
│   ├── train.py         # 模型训练入口
│   ├── evaluate.py      # 模型评估入口
│   └── utils.py         # 模型创建、保存、加载工具函数
├── server/               # HTTP 服务器（用于外部控制）
│   ├── server.py        # HTTP API 服务器实现
│   └── start_server.sh  # 服务器启动脚本
├── script/               # 其他脚本
├── requirements.txt      # Python 依赖列表
└── README.md            # 本文档
```

## 使用方法

### 1. 训练模型

训练 DDPG 模型来自动调优 Broker 配置：

```bash
cd /home/qincai/userDir/BrokerTuner
export PYTHONPATH=$(pwd):$PYTHONPATH
export MOSQUITTO_PID=$(pgrep mosquitto)

python -m tuner.train \
    --total-timesteps 100000 \
    --save-dir ./checkpoints \
    --save-freq 10000 \
    --device cpu
```

**参数说明：**
- `--total-timesteps`: 总训练步数（默认：100000）
- `--save-dir`: 模型保存目录（默认：./checkpoints）
- `--save-freq`: 每隔多少步保存一次 checkpoint（默认：10000）
- `--device`: 训练设备，`cpu` 或 `cuda`（默认：cpu）

**训练过程：**
- 模型会定期保存 checkpoint 到 `--save-dir` 目录
- 训练完成后会保存最终模型为 `ddpg_mosquitto_final.zip`
- 训练过程中会实时应用配置到 Mosquitto Broker（需要相应权限）

### 2. 评估模型

评估已训练好的模型性能：

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export MOSQUITTO_PID=$(pgrep mosquitto)

python -m tuner.evaluate \
    --model-path ./checkpoints/ddpg_mosquitto_final.zip \
    --n-episodes 10 \
    --device cpu
```

**参数说明：**
- `--model-path`: 模型文件路径（.zip 文件）
- `--n-episodes`: 评估的 episode 数量（默认：10）
- `--device`: 评估设备，`cpu` 或 `cuda`（默认：cpu）

**输出示例：**
```
Episode 1/10: reward = 1.234
Episode 2/10: reward = 1.456
...
========================================
Mean reward over 10 episodes: 1.345
Std reward: 0.123
```

### 3. 启动配置服务器（可选）

启动 HTTP 服务器，允许外部系统通过 API 调整 Broker 配置：

```bash
cd /home/qincai/userDir/BrokerTuner
bash server/start_server.sh
```

服务器默认监听 `0.0.0.0:8080`。

**API 使用示例：**

```bash
curl -X POST http://localhost:8080/apply_knobs \
  -H "Content-Type: application/json" \
  -d '{
    "max_inflight_messages": 100,
    "max_inflight_bytes": 0,
    "max_queued_messages": 2000,
    "max_queued_bytes": 0,
    "queue_qos0_messages": true,
    "memory_limit": 0,
    "persistence": false,
    "autosave_interval": 0,
    "set_tcp_nodelay": true,
    "max_packet_size": 1048576,
    "message_size_limit": 1048576
  }'
```

**响应示例：**
```json
{"status": "ok"}
```

### 4. 在代码中使用环境

```python
from environment import MosquittoBrokerEnv, EnvConfig
from tuner.utils import make_ddpg_model, make_env

# 创建环境
env_cfg = EnvConfig()
env = make_env(env_cfg)

# 创建模型
model = make_ddpg_model(env, device="cpu")

# 训练
model.learn(total_timesteps=100000)

# 评估
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

## 配置说明

### 环境配置

在 `environment/config.py` 中可以调整：

- **MQTT 配置** (`MQTTConfig`):
  - `host`: Mosquitto Broker 地址（默认：127.0.0.1）
  - `port`: MQTT 端口（默认：1883）
  - `timeout_sec`: 采样超时时间（默认：2.0 秒）

- **进程配置** (`ProcConfig`):
  - `pid`: Mosquitto 进程 PID（通过环境变量 `MOSQUITTO_PID` 设置）
  - `cpu_norm`: CPU 归一化参考值（默认：400.0）
  - `mem_norm`: 内存归一化参考值（默认：1 GiB）
  - `ctxt_norm`: 上下文切换归一化参考值（默认：1e6）

- **环境配置** (`EnvConfig`):
  - `step_interval_sec`: 每步动作后等待系统稳定的时间（默认：1.0 秒）
  - `max_steps`: 每个 episode 的最大步数（默认：200）
  - `state_dim`: 状态向量维度（默认：5）
  - `action_dim`: 动作向量维度（默认：11）

### 模型配置

在 `tuner/utils.py` 的 `make_ddpg_model` 函数中可以调整：

- `learning_rate`: 学习率（默认：1e-4）
- `batch_size`: 批次大小（默认：128）
- `gamma`: 折扣因子（默认：0.99）
- `tau`: 软更新系数（默认：0.005）

### Reward 函数

当前 reward 函数在 `environment/broker.py` 的 `_compute_reward` 方法中定义：

```python
reward = α * clients_norm + β * msg_rate_norm - γ * cpu_ratio - δ * mem_ratio
```

可以根据实际需求修改权重系数或添加时延约束项。

## 状态向量说明

当前状态向量包含 5 个维度：

1. **clients_norm**: 归一化的连接数（`$SYS/broker/clients/connected / 1000`）
2. **msg_rate_norm**: 归一化的消息速率（`$SYS/broker/messages/received / 10000`）
3. **cpu_ratio**: CPU 使用占比（从 `/proc/[pid]/stat` 计算）
4. **mem_ratio**: 内存使用占比（从 `/proc/[pid]/status` 的 VmRSS 计算）
5. **ctxt_ratio**: 上下文切换占比（从 `/proc/[pid]/status` 计算）

## 常见问题

### 1. 权限问题

**问题**: `apply_knobs` 失败，提示权限不足

**解决**: 
- 确保运行脚本的用户有权限修改 `/etc/mosquitto/conf.d/` 目录
- 确保有权限执行 `systemctl reload/restart mosquitto`
- 可以使用 `sudo` 运行训练脚本，或配置 sudoers 允许无密码执行相关命令

### 2. Mosquitto PID 未找到

**问题**: `ProcConfig.pid 未设置或非法`

**解决**:
```bash
export MOSQUITTO_PID=$(pgrep mosquitto)
# 或手动指定
export MOSQUITTO_PID=12345
```

### 3. MQTT 连接失败

**问题**: 无法连接到 Mosquitto Broker

**解决**:
- 检查 Mosquitto 是否正在运行：`systemctl status mosquitto`
- 检查 `environment/config.py` 中的 `MQTTConfig` 配置是否正确
- 检查防火墙设置

### 4. 配置文件写入失败

**问题**: 无法写入 `/etc/mosquitto/conf.d/broker_tuner.conf`

**解决**:
- 确保目录存在：`sudo mkdir -p /etc/mosquitto/conf.d`
- 检查权限：`ls -la /etc/mosquitto/conf.d/`
- 使用环境变量指定其他路径：`export MOSQUITTO_TUNER_CONFIG=/path/to/custom.conf`

### 5. 使用 GPU 训练

**问题**: 希望使用 GPU 加速训练

**解决**:
- 确保已安装 CUDA 版本的 PyTorch
- 训练时指定设备：`--device cuda`
- 评估时指定设备：`python -m tuner.evaluate --device cuda ...`

### 6. 状态采样为空

**问题**: `$SYS/#` 主题没有收到消息

**解决**:
- 检查 Mosquitto 配置中是否启用了 `$SYS` 主题：在 `mosquitto.conf` 中添加 `sys_interval 10`
- 检查订阅的主题是否正确：默认订阅 `$SYS/#`

## 开发说明

### 自定义 Reward 函数

修改 `environment/broker.py` 中的 `_compute_reward` 方法：

```python
def _compute_reward(self, prev_state, next_state):
    # 你的自定义 reward 计算逻辑
    # 例如：加入时延约束
    latency = self._get_latency()  # 需要实现
    if latency > LATENCY_THRESHOLD:
        return -10.0  # 严重惩罚
    # ... 其他计算
    return reward
```

### 扩展状态向量

修改 `environment/utils.py` 中的 `build_state_vector` 函数，添加更多指标。

### 添加新的配置参数

1. 在 `environment/knobs.py` 的 `BrokerKnobSpace` 中添加新的范围定义
2. 在 `decode_action` 方法中添加映射逻辑
3. 在 `apply_knobs` 函数中添加配置写入逻辑
4. 更新 `EnvConfig.action_dim`

## 许可证

[根据项目实际情况填写]

## 联系方式

[根据项目实际情况填写]
