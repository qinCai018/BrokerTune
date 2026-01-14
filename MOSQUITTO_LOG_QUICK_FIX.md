# Mosquitto日志快速修复指南

## 🚨 立即清理42GB日志

```bash
cd /home/qincai/userDir/BrokerTuner

# 立即清理旧日志（只保留最新的3个）
sudo ./script/cleanup_mosquitto_logs.sh
```

## ⚙️ 配置日志控制（防止再次出现）

### 步骤1：配置Mosquitto日志级别

```bash
sudo ./script/configure_mosquitto_logging.sh
```

这会：
- 设置日志级别为warning（减少日志输出）
- 配置logrotate自动管理日志
- 只保留最近3个日志文件

### 步骤2：训练时启用自动清理

```bash
./script/run_train.sh \
    --enable-workload \
    --total-timesteps 100000 \
    --save-dir ./checkpoints \
    --save-freq 10000 \
    --cleanup-mosquitto-logs \
    --mosquitto-log-cleanup-freq 5000 \
    --max-mosquitto-log-files 3
```

## 📊 检查日志大小

```bash
# 查看日志目录大小
sudo du -sh /var/log/mosquitto

# 查看各个日志文件
sudo ls -lh /var/log/mosquitto/
```

## 🔧 手动清理（如果需要）

```bash
# 清理所有旧日志（只保留最新的3个）
sudo ./script/cleanup_mosquitto_logs.sh

# 或者手动删除
sudo find /var/log/mosquitto -name "*.log.*.gz" -type f | \
    sudo xargs ls -t | tail -n +4 | sudo xargs rm -f
```

## 📚 详细文档

更多信息请参考: `docs/training/MOSQUITTO_LOG_CONTROL.md`
