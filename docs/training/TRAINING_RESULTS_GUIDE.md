# 训练结果使用指南

## 一、训练结果位置

训练完成后，所有结果保存在 `--save-dir` 目录（默认：`./checkpoints/`）中。

### 目录结构
```
checkpoints/
├── ddpg_mosquitto_500_steps.zip          # 第500步的checkpoint
├── ddpg_mosquitto_1000_steps.zip         # 第1000步的checkpoint
├── ddpg_mosquitto_final.zip              # 最终训练好的模型 ⭐
├── ddpg_mosquitto_replay_buffer_500_steps.pkl   # 经验回放缓冲区
├── ddpg_mosquitto_replay_buffer_1000_steps.pkl
├── logs/                                 # 训练日志
│   ├── progress.csv                      # CSV格式的训练指标
│   └── events.out.tfevents.*            # TensorBoard日志（如果安装了tensorboard）
└── monitor/                              # Episode统计
    └── monitor.csv                      # 每个episode的奖励和长度
```

## 二、训练结果内容

### 1. 模型文件（.zip）⭐
- **内容**：训练好的 DDPG 神经网络模型
  - Actor 网络：根据当前状态输出最优动作（配置参数）
  - Critic 网络：评估状态-动作对的 Q 值
  - 目标网络：用于稳定训练
  - 优化器状态：用于继续训练
- **用途**：用于推理（获取最优配置）或继续训练

### 2. 训练日志（logs/progress.csv）
- **内容**：训练过程中的各种指标
  - `time_elapsed`: 训练时间
  - `total_timesteps`: 总步数
  - `rollout/ep_rew_mean`: 平均episode奖励
  - `rollout/ep_len_mean`: 平均episode长度
  - `train/actor_loss`: Actor损失
  - `train/critic_loss`: Critic损失
  - `train/learning_rate`: 学习率
- **用途**：分析训练过程，判断模型是否收敛

### 3. Monitor日志（monitor/monitor.csv）
- **内容**：每个episode的详细统计
  - `r`: episode总奖励
  - `l`: episode长度（步数）
  - `t`: episode时间戳
- **用途**：分析episode级别的性能

### 4. 经验回放缓冲区（.pkl）
- **内容**：训练过程中的经验数据
- **用途**：继续训练时使用，或用于离线分析

## 三、如何使用训练结果

### 方法1：评估模型性能

评估训练好的模型在多个episode上的平均表现：

```bash
cd /home/qincai/userDir/BrokerTuner

./script/run_evaluate.sh \
    --model-path ./checkpoints/ddpg_mosquitto_final.zip \
    --n-episodes 10
```

**输出示例：**
```
Episode 1/10: reward = 1.234
Episode 2/10: reward = 1.456
...
========================================
Mean reward over 10 episodes: 1.345
Std reward: 0.123
```

### 方法2：获取并应用最优配置 ⭐⭐⭐

使用训练好的模型，根据当前Broker状态获取最优配置：

```bash
cd /home/qincai/userDir/BrokerTuner

# 1. 查看模型推荐的最优配置（不应用）
python3 script/apply_optimal_config.py \
    --model-path ./checkpoints/ddpg_mosquitto_final.zip \
    --dry-run

# 2. 获取最优配置并应用到Broker（需要sudo权限）
sudo python3 script/apply_optimal_config.py \
    --model-path ./checkpoints/ddpg_mosquitto_final.zip \
    --apply-config
```

**输出示例：**
```
当前 Broker 状态:
  连接数（归一化）: 0.1000
  消息速率（归一化）: 0.6666
  CPU 使用率: 0.2500
  内存使用率: 0.1500
  上下文切换率: 0.0500

模型推荐的最优配置:
  max_inflight_messages: 796
  max_inflight_bytes: 38695812
  max_queued_messages: 6898
  ...
```

### 方法3：在代码中使用模型

```python
from tuner.utils import load_model, make_env
from environment import EnvConfig

# 1. 创建环境和加载模型
env_cfg = EnvConfig()
env = make_env(env_cfg)
model = load_model("./checkpoints/ddpg_mosquitto_final.zip", env, device="cpu")

# 2. 获取当前状态
reset_result = env.reset()
if isinstance(reset_result, tuple):
    obs, _ = reset_result
else:
    obs = reset_result

# 3. 使用模型预测最优动作（配置）
action, _ = model.predict(obs, deterministic=True)

# 4. 将动作解码为配置参数
knobs = env.knob_space.decode_action(action)

print("最优配置:", knobs)

# 5. 应用配置（需要权限）
from environment.knobs import apply_knobs
apply_knobs(knobs)

env.close()
```

### 方法4：继续训练

从checkpoint继续训练：

```python
from tuner.utils import load_model, make_env

env = make_env()
model = load_model("./checkpoints/ddpg_mosquitto_1000_steps.zip", env, device="cpu")

# 继续训练
model.learn(total_timesteps=50000)  # 再训练50000步

# 保存新模型
model.save("./checkpoints/ddpg_mosquitto_1500_steps")
```

### 方法5：分析训练过程

#### 查看CSV日志
```bash
# 查看训练指标
head -20 ./checkpoints/logs/progress.csv

# 使用Python分析
python3 << EOF
import pandas as pd
df = pd.read_csv('./checkpoints/logs/progress.csv')
print('平均奖励:', df['rollout/ep_rew_mean'].mean())
print('最大奖励:', df['rollout/ep_rew_mean'].max())
print('最后10个episode平均奖励:', df['rollout/ep_rew_mean'].tail(10).mean())
EOF
```

#### 使用TensorBoard可视化
```bash
# 如果安装了tensorboard
tensorboard --logdir ./checkpoints/logs

# 然后在浏览器中打开 http://localhost:6006
```

## 四、训练结果的含义

### 模型输出的是什么？

训练好的模型是一个**策略网络**，它学习到了：
- **输入**：当前Broker状态（连接数、消息速率、CPU、内存等）
- **输出**：最优的配置参数（11个参数的组合）

### 模型如何工作？

1. **观察状态**：模型读取当前Broker的运行状态
2. **预测动作**：根据学习到的策略，输出最优的配置参数
3. **应用配置**：将配置应用到Broker
4. **评估效果**：通过奖励函数评估配置的效果

### 最优配置的含义

模型输出的配置参数是**针对当前工作负载和系统状态优化的**：
- 如果工作负载变化，最优配置也会变化
- 模型会根据实时状态动态调整配置
- 目标是最大化性能（消息吞吐量）同时最小化资源消耗（CPU、内存）

## 五、实际应用场景

### 场景1：生产环境部署

```bash
# 1. 训练模型（在测试环境）
./script/run_train.sh --total-timesteps 1000000 --enable-workload

# 2. 在生产环境应用最优配置
sudo python3 script/apply_optimal_config.py \
    --model-path ./checkpoints/ddpg_mosquitto_final.zip \
    --apply-config
```

### 场景2：持续优化

定期运行模型，根据当前负载动态调整配置：

```bash
# 每小时运行一次，获取并应用最优配置
*/60 * * * * cd /path/to/BrokerTuner && \
    sudo python3 script/apply_optimal_config.py \
    --model-path ./checkpoints/ddpg_mosquitto_final.zip \
    --apply-config
```

### 场景3：A/B测试

比较不同配置的效果：

```python
# 使用模型推荐的配置
optimal_knobs = get_optimal_config(model, env)
apply_knobs(optimal_knobs)

# 运行一段时间后，与默认配置对比
# 记录性能指标（吞吐量、延迟、资源使用）
```

## 六、检查训练质量

### 1. 查看训练日志
```bash
# 检查奖励是否提升
tail -20 ./checkpoints/logs/progress.csv
```

### 2. 评估模型性能
```bash
# 运行评估，查看平均奖励
python3 -m tuner.evaluate \
    --model-path ./checkpoints/ddpg_mosquitto_final.zip \
    --n-episodes 20
```

### 3. 对比不同checkpoint
```bash
# 评估不同训练阶段的模型
for model in ./checkpoints/ddpg_mosquitto_*_steps.zip; do
    echo "评估: $model"
    python3 -m tuner.evaluate --model-path "$model" --n-episodes 5
done
```

## 七、常见问题

### Q1: 如何知道模型训练好了？
**A:** 查看训练日志：
- 奖励曲线是否上升并趋于稳定
- Episode长度是否合理
- 损失是否下降

### Q2: 模型输出的配置一定是最优的吗？
**A:** 不一定。模型的质量取决于：
- 训练步数是否足够
- 工作负载是否真实
- 奖励函数是否合理
- 超参数是否调优

### Q3: 如何验证配置效果？
**A:** 
1. 应用配置前后对比性能指标
2. 使用 `mosquitto_sub` 监听 `$SYS/#` 主题查看Broker指标
3. 监控CPU、内存使用率
4. 测量消息吞吐量和延迟

### Q4: 模型可以用于不同的工作负载吗？
**A:** 
- 可以，但效果可能不如针对特定工作负载训练的模型
- 建议为不同工作负载训练不同的模型
- 或者使用迁移学习技术

## 八、总结

训练结果包括：
- ✅ **模型文件**：用于获取最优配置
- ✅ **训练日志**：用于分析训练过程
- ✅ **Episode统计**：用于评估模型性能

使用方法：
1. **评估模型**：`python3 -m tuner.evaluate`
2. **应用配置**：`python3 script/apply_optimal_config.py --apply-config`
3. **继续训练**：从checkpoint加载模型继续训练
4. **分析日志**：查看CSV或TensorBoard日志

训练好的模型是一个**智能配置优化器**，可以根据当前Broker状态自动推荐最优配置参数！
