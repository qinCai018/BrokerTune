# 训练磁盘空间优化指南

## 概述

训练过程中会产生大量文件，占用大量磁盘空间。本指南介绍如何使用优化参数来减少磁盘占用。

## 磁盘占用分析

训练过程中主要产生以下文件：

1. **Checkpoint文件** (`.zip`) - 每个约3.9MB
   - 包含模型权重和配置
   - 默认每10,000步保存一次

2. **Replay Buffer文件** (`.pkl`) - 每个约92MB ⚠️ **最大占用**
   - 包含经验回放缓冲区数据
   - 用于恢复训练状态

3. **TensorBoard日志** (`events.out.tfevents.*`) - 每个约4KB
   - 训练指标可视化数据
   - 会累积大量文件

4. **CSV日志文件**
   - `action_throughput_log.csv` - Action和吞吐量记录
   - `logs/progress.csv` - 训练进度指标
   - `monitor/monitor.csv` - Episode统计

## 优化参数

### 1. 不保存Replay Buffer（推荐）

**节省空间**: 每个checkpoint节省约92MB

```bash
./script/run_train.sh \
    --enable-workload \
    --total-timesteps 100000 \
    --save-dir ./checkpoints \
    --save-freq 10000
    # 默认不保存replay buffer
```

如果需要恢复训练，可以启用：

```bash
--save-replay-buffer  # 启用replay buffer保存
```

**注意**: 不保存replay buffer意味着无法完全恢复训练状态，但可以加载模型权重继续训练。

### 2. 限制Checkpoint数量

**节省空间**: 自动删除旧的checkpoint，只保留最新的N个

```bash
./script/run_train.sh \
    --enable-workload \
    --total-timesteps 100000 \
    --save-dir ./checkpoints \
    --save-freq 10000 \
    --max-checkpoints 3  # 只保留最新的3个checkpoint
```

**工作原理**: 
- 每次保存checkpoint后，自动检查checkpoint数量
- 如果超过`--max-checkpoints`，删除最旧的checkpoint和对应的replay buffer
- 默认值：3个checkpoint

### 3. 禁用TensorBoard日志

**节省空间**: 减少日志文件数量

```bash
./script/run_train.sh \
    --enable-workload \
    --total-timesteps 100000 \
    --save-dir ./checkpoints \
    --save-freq 10000 \
    --disable-tensorboard  # 禁用TensorBoard日志
```

**注意**: 禁用后无法使用TensorBoard可视化训练过程，但仍会保存CSV日志。

### 4. 限制Action日志记录频率

**节省空间**: 减少CSV文件大小

```bash
./script/run_train.sh \
    --enable-workload \
    --total-timesteps 100000 \
    --save-dir ./checkpoints \
    --save-freq 10000 \
    --limit-action-log \
    --action-log-interval 10  # 每10步记录一次
```

**工作原理**:
- 默认每步都记录action和吞吐量
- 启用`--limit-action-log`后，只记录每N步（`--action-log-interval`）
- 前3步总是记录（用于调试）

## 完整优化示例

### 最小磁盘占用配置

```bash
./script/run_train.sh \
    --enable-workload \
    --total-timesteps 100000 \
    --save-dir ./checkpoints \
    --save-freq 20000 \
    --max-checkpoints 2 \
    --disable-tensorboard \
    --limit-action-log \
    --action-log-interval 20
```

**预期磁盘占用**:
- Checkpoint: 2个 × 3.9MB = 7.8MB
- Replay Buffer: 0MB（不保存）
- CSV日志: ~1-5MB（取决于训练步数）
- **总计**: 约10-15MB（相比默认配置节省约200MB）

### 平衡配置（推荐）

```bash
./script/run_train.sh \
    --enable-workload \
    --total-timesteps 100000 \
    --save-dir ./checkpoints \
    --save-freq 10000 \
    --max-checkpoints 3 \
    --limit-action-log \
    --action-log-interval 10
```

**预期磁盘占用**:
- Checkpoint: 3个 × 3.9MB = 11.7MB
- Replay Buffer: 0MB（不保存）
- CSV日志: ~5-10MB
- TensorBoard: ~100KB
- **总计**: 约20-25MB

## 参数对比表

| 参数 | 默认值 | 优化值 | 节省空间 |
|------|--------|--------|----------|
| `--save-replay-buffer` | False | False | 每个checkpoint节省92MB |
| `--max-checkpoints` | 3 | 2-3 | 自动清理旧文件 |
| `--disable-tensorboard` | False | True | 减少日志文件 |
| `--limit-action-log` | False | True | 减少CSV大小 |
| `--action-log-interval` | N/A | 10-20 | 减少CSV大小 |
| `--save-freq` | 10000 | 20000 | 减少checkpoint数量 |

## 磁盘空间监控

训练过程中监控磁盘使用：

```bash
# 查看checkpoints目录大小
du -sh ./checkpoints

# 查看各个文件大小
du -sh ./checkpoints/*

# 查看磁盘使用情况
df -h /home/qincai/userDir/BrokerTuner
```

## 自动清理

训练脚本已集成自动清理功能：

1. **Checkpoint自动清理**: 每次保存checkpoint后自动删除超出数量的旧文件
2. **Replay Buffer清理**: 删除checkpoint时同时删除对应的replay buffer文件

## 手动清理

如果需要手动清理：

```bash
# 删除所有checkpoint（保留最新的）
ls -t ./checkpoints/ddpg_mosquitto_*_steps.zip | tail -n +2 | xargs rm -f

# 删除所有replay buffer
rm -f ./checkpoints/*replay_buffer*.pkl

# 删除TensorBoard日志
rm -f ./checkpoints/logs/events.out.tfevents.*

# 使用清理脚本
./cleanup_training.sh
```

## 最佳实践

1. **开发/测试阶段**: 使用最小磁盘占用配置
2. **正式训练**: 使用平衡配置，保留必要的checkpoint和日志
3. **长期训练**: 
   - 增加`--save-freq`（如50000）
   - 减少`--max-checkpoints`（如2）
   - 启用`--limit-action-log`

## 注意事项

1. **不保存Replay Buffer**: 无法完全恢复训练状态，但可以加载模型权重继续训练
2. **限制日志频率**: 可能丢失部分训练细节，但保留关键信息
3. **自动清理**: 确保有足够的磁盘空间，避免清理过程中磁盘满导致失败
4. **备份重要checkpoint**: 定期备份重要的checkpoint到其他位置

## 故障排除

### 问题：磁盘空间不足

**解决方案**:
1. 立即停止训练
2. 清理旧的checkpoint和日志
3. 使用优化参数重新启动训练

### 问题：需要恢复训练但replay buffer被删除

**解决方案**:
1. 加载最新的checkpoint模型权重
2. 重新初始化replay buffer（会从头开始收集经验）
3. 继续训练

### 问题：日志文件过大

**解决方案**:
1. 启用`--limit-action-log`和`--action-log-interval`
2. 定期清理旧的日志文件
3. 考虑只保留最近的日志
