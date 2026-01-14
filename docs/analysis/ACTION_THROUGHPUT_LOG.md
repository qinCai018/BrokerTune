# Action和吞吐量日志说明

## 功能概述

训练过程中会自动记录每一步的：
- **Action（动作）**：11维动作向量，表示Broker配置参数
- **Throughput（吞吐量）**：消息速率归一化值（从状态向量提取）
- **Reward（奖励）**：当前步的奖励值
- **Step（步数）**：当前步数
- **Episode（回合）**：当前episode编号

## 日志文件位置

训练完成后，日志文件保存在：
```
{save_dir}/action_throughput_log.csv
```

例如，如果使用 `--save-dir ./checkpoints`，则日志文件在：
```
./checkpoints/action_throughput_log.csv
```

## CSV文件格式

CSV文件包含以下列：

| 列名 | 说明 | 示例值 |
|------|------|--------|
| `step` | 当前步数（episode内） | 1, 2, 3, ... |
| `episode` | Episode编号 | 1, 2, 3, ... |
| `action_0_max_inflight_messages` | 动作0：最大飞行中消息数 | 0.5 |
| `action_1_max_inflight_bytes` | 动作1：最大飞行中字节数 | 0.3 |
| `action_2_max_queued_messages` | 动作2：最大排队消息数 | 0.7 |
| `action_3_max_queued_bytes` | 动作3：最大排队字节数 | 0.4 |
| `action_4_queue_qos0_messages` | 动作4：是否排队QoS0消息 | 0.8 (>=0.5为True) |
| `action_5_memory_limit` | 动作5：内存限制 | 0.6 |
| `action_6_persistence` | 动作6：是否持久化 | 0.9 (>=0.5为True) |
| `action_7_autosave_interval` | 动作7：自动保存间隔 | 0.2 |
| `action_8_set_tcp_nodelay` | 动作8：是否设置TCP_NODELAY | 0.7 (>=0.5为True) |
| `action_9_max_packet_size` | 动作9：最大数据包大小 | 0.5 |
| `action_10_message_size_limit` | 动作10：消息大小限制 | 0.5 |
| `throughput` | 吞吐量（消息速率归一化值） | 0.6666 |
| `reward` | 奖励值 | 1.234 |

## 使用方法

### 1. 开始训练（自动记录）

训练时会自动记录，无需额外配置：

```bash
./script/run_train.sh \
    --total-timesteps 10000 \
    --save-dir ./checkpoints \
    --enable-workload
```

### 2. 查看日志文件

训练完成后，查看日志：

```bash
# 查看前20行
head -20 ./checkpoints/action_throughput_log.csv

# 查看总行数（包括表头）
wc -l ./checkpoints/action_throughput_log.csv

# 使用Python分析
python3 << EOF
import pandas as pd

# 读取CSV文件
df = pd.read_csv('./checkpoints/action_throughput_log.csv')

# 显示基本信息
print(f"总步数: {len(df)}")
print(f"总Episode数: {df['episode'].max()}")
print(f"\n吞吐量统计:")
print(f"  平均值: {df['throughput'].mean():.4f}")
print(f"  最大值: {df['throughput'].max():.4f}")
print(f"  最小值: {df['throughput'].min():.4f}")
print(f"\n奖励统计:")
print(f"  平均值: {df['reward'].mean():.4f}")
print(f"  最大值: {df['reward'].max():.4f}")
print(f"  最小值: {df['reward'].min():.4f}")

# 显示每个episode的平均吞吐量
print(f"\n每个Episode的平均吞吐量:")
episode_stats = df.groupby('episode')['throughput'].mean()
print(episode_stats.head(10))
EOF
```

### 3. 使用Excel或Pandas分析

#### 使用Pandas分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('./checkpoints/action_throughput_log.csv')

# 1. 绘制吞吐量随时间的变化
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['throughput'], label='Throughput')
plt.xlabel('Step')
plt.ylabel('Throughput (normalized)')
plt.title('Throughput Over Time')
plt.legend()
plt.grid(True)
plt.savefig('throughput_over_time.png')

# 2. 绘制奖励随时间的变化
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['reward'], label='Reward', color='green')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Reward Over Time')
plt.legend()
plt.grid(True)
plt.savefig('reward_over_time.png')

# 3. 分析action与吞吐量的关系
# 选择某个action维度进行分析
action_col = 'action_0_max_inflight_messages'
plt.figure(figsize=(12, 6))
plt.scatter(df[action_col], df['throughput'], alpha=0.5)
plt.xlabel(action_col)
plt.ylabel('Throughput')
plt.title(f'Relationship between {action_col} and Throughput')
plt.grid(True)
plt.savefig('action_vs_throughput.png')

# 4. 每个episode的平均吞吐量
episode_throughput = df.groupby('episode')['throughput'].mean()
plt.figure(figsize=(12, 6))
plt.plot(episode_throughput.index, episode_throughput.values)
plt.xlabel('Episode')
plt.ylabel('Average Throughput')
plt.title('Average Throughput per Episode')
plt.grid(True)
plt.savefig('episode_throughput.png')
```

#### 使用Excel分析

1. 打开Excel
2. 导入CSV文件：`数据` → `从文本/CSV导入` → 选择 `action_throughput_log.csv`
3. 使用Excel的数据分析功能：
   - 创建数据透视表分析不同action值对应的吞吐量
   - 使用图表功能绘制吞吐量和奖励的变化曲线
   - 使用条件格式高亮显示高吞吐量的行

## 数据说明

### Action值范围

所有action值都在 `[0, 1]` 范围内：
- `0.0` 到 `1.0`：归一化的动作值
- 对于布尔类型的action（如 `action_4`, `action_6`, `action_8`）：
  - `>= 0.5`：True
  - `< 0.5`：False

### 吞吐量说明

- **来源**：从状态向量的第1维提取（`state[1]`）
- **含义**：消息速率归一化值
- **计算方式**：`messages_received / 10000.0`
- **范围**：通常是 `[0, 1]` 或更大（取决于实际消息速率）

### 奖励说明

- **来源**：环境的奖励函数计算
- **含义**：性能改进奖励 - 资源惩罚
- **公式**：`reward = α * (D_t - D_{t-1}) + β * (D_t - D_0) - 资源惩罚`
  - `α = 10.0`：短期改进权重
  - `β = 5.0`：长期改进权重

## 常见用途

### 1. 分析最优配置

找出吞吐量最高的action组合：

```python
import pandas as pd

df = pd.read_csv('./checkpoints/action_throughput_log.csv')

# 找出吞吐量最高的10行
top_throughput = df.nlargest(10, 'throughput')
print("吞吐量最高的10个配置:")
print(top_throughput[['step', 'episode'] + [f'action_{i}' for i in range(11)] + ['throughput', 'reward']])
```

### 2. 分析action与性能的关系

```python
import pandas as pd
import numpy as np

df = pd.read_csv('./checkpoints/action_throughput_log.csv')

# 分析每个action维度与吞吐量的相关性
action_cols = [f'action_{i}' for i in range(11)]
correlations = df[action_cols + ['throughput']].corr()['throughput'].sort_values(ascending=False)
print("Action与吞吐量的相关性:")
print(correlations)
```

### 3. 训练过程可视化

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./checkpoints/action_throughput_log.csv')

# 绘制训练曲线
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 吞吐量曲线
axes[0].plot(df.index, df['throughput'], label='Throughput', alpha=0.7)
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Throughput')
axes[0].set_title('Throughput Over Training Steps')
axes[0].legend()
axes[0].grid(True)

# 奖励曲线
axes[1].plot(df.index, df['reward'], label='Reward', color='green', alpha=0.7)
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Reward')
axes[1].set_title('Reward Over Training Steps')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
```

## 注意事项

1. **文件大小**：如果训练步数很多（如100万步），CSV文件可能很大（几十MB到几百MB）
2. **实时写入**：数据是实时写入CSV文件的，训练过程中可以随时查看
3. **Episode编号**：每个episode从1开始计数，step在每个episode内从1开始计数
4. **吞吐量单位**：吞吐量是归一化值，不是实际的消息/秒。要转换为实际值，需要乘以10000

## 示例输出

训练完成后会看到类似输出：

```
[ActionThroughputLogger] 已记录 200 步数据（episode 1）
[ActionThroughputLogger] 数据已保存到: ./checkpoints/action_throughput_log.csv

✅ Action和吞吐量日志已保存到: ./checkpoints/action_throughput_log.csv
   可以使用以下命令查看:
   head -20 ./checkpoints/action_throughput_log.csv
   或使用Excel/Pandas打开CSV文件进行分析
```

## 相关文件

- `tuner/train.py`：包含 `ActionThroughputLoggerWrapper` 类的实现
- `environment/broker.py`：环境实现，包含状态采样和奖励计算
- `environment/utils.py`：包含状态向量构建函数
