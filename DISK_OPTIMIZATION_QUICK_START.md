# 磁盘空间优化快速指南

## 🎯 快速开始

### 最小磁盘占用训练（推荐用于测试）

```bash
cd /home/qincai/userDir/BrokerTuner

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

**预期磁盘占用**: 约10-15MB（相比默认配置节省约200MB）

### 平衡配置（推荐用于正式训练）

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

**预期磁盘占用**: 约20-25MB

## 📊 优化参数说明

| 参数 | 作用 | 节省空间 |
|------|------|----------|
| `--max-checkpoints 3` | 只保留最新的3个checkpoint | 自动清理旧文件 |
| `--disable-tensorboard` | 禁用TensorBoard日志 | 减少日志文件 |
| `--limit-action-log` | 限制action日志频率 | 减少CSV大小 |
| `--action-log-interval 10` | 每10步记录一次 | 减少CSV大小 |
| 默认不保存replay buffer | 不保存92MB的replay buffer | 每个checkpoint节省92MB |

## 🔍 监控磁盘使用

```bash
# 查看checkpoints目录大小
du -sh ./checkpoints

# 查看各个文件大小
du -sh ./checkpoints/*

# 查看磁盘使用情况
df -h
```

## 📚 详细文档

更多信息请参考: `docs/training/DISK_SPACE_OPTIMIZATION.md`
