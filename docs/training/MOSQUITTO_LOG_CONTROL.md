# Mosquitto日志控制指南

## 问题描述

训练过程中，Mosquitto会产生大量日志文件，占用大量磁盘空间（可能达到几十GB）。

## 解决方案

### 方案1：配置Mosquitto日志级别（推荐）

减少日志输出，只记录警告和错误：

```bash
cd /home/qincai/userDir/BrokerTuner

# 运行配置脚本
sudo ./script/configure_mosquitto_logging.sh
```

**功能**：
- 设置日志级别为`warning`（只记录警告和错误）
- 配置logrotate自动轮转日志
- 只保留最近3个日志文件
- 自动压缩旧日志

**效果**：大幅减少日志文件大小和数量

### 方案2：在训练时启用自动清理

训练脚本已集成Mosquitto日志清理功能：

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

**参数说明**：
- `--cleanup-mosquitto-logs`: 启用Mosquitto日志自动清理
- `--mosquitto-log-cleanup-freq N`: 每隔N步清理一次（默认：5000）
- `--max-mosquitto-log-files N`: 最多保留N个日志文件（默认：3）

**工作原理**：
- 定期检查日志文件数量
- 删除超出数量的旧日志文件
- 如果当前日志文件超过100MB，自动清空

### 方案3：手动清理日志

```bash
# 清理Mosquitto日志
sudo ./script/cleanup_mosquitto_logs.sh
```

**功能**：
- 删除旧的压缩日志文件（只保留最新的3个）
- 如果当前日志文件超过100MB，清空它
- 显示清理前后的磁盘使用情况

## 完整配置示例

### 步骤1：配置Mosquitto日志（一次性配置）

```bash
sudo ./script/configure_mosquitto_logging.sh
```

### 步骤2：启动训练（启用自动清理）

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

### 步骤3：定期手动清理（可选）

如果需要立即清理，可以运行：

```bash
sudo ./script/cleanup_mosquitto_logs.sh
```

## 日志配置详解

### Mosquitto日志级别

| 级别 | 说明 | 推荐用途 |
|------|------|----------|
| `error` | 只记录错误 | 最小日志输出 |
| `warning` | 记录警告和错误 | **推荐**（平衡） |
| `notice` | 记录通知、警告和错误 | 正常使用 |
| `information` | 记录所有信息 | 调试（会产生大量日志） |
| `all` | 记录所有日志 | 调试（会产生大量日志） |

### Logrotate配置

创建的logrotate配置：
- **轮转频率**: 每天
- **保留文件数**: 3个
- **压缩**: 自动压缩旧日志
- **触发**: 自动触发，无需手动操作

## 监控日志大小

```bash
# 查看Mosquitto日志目录大小
sudo du -sh /var/log/mosquitto

# 查看各个日志文件大小
sudo du -sh /var/log/mosquitto/*

# 查看日志文件数量
sudo ls -lh /var/log/mosquitto/ | wc -l
```

## 故障排除

### 问题1：配置脚本需要sudo权限

**解决方案**：
```bash
# 确保脚本有执行权限
chmod +x ./script/configure_mosquitto_logging.sh

# 使用sudo运行
sudo ./script/configure_mosquitto_logging.sh
```

### 问题2：日志清理失败（权限不足）

**说明**：训练脚本中的日志清理需要sudo权限，如果失败会静默忽略。

**解决方案**：
1. 手动运行清理脚本：`sudo ./script/cleanup_mosquitto_logs.sh`
2. 或者配置无密码sudo（不推荐，安全风险）

### 问题3：日志仍然快速增长

**可能原因**：
1. 日志级别设置未生效
2. Logrotate未正确配置
3. 训练过程中产生大量错误/警告

**解决方案**：
1. 检查日志配置：`sudo cat /etc/mosquitto/conf.d/broker_tuner_logging.conf`
2. 检查logrotate配置：`sudo cat /etc/logrotate.d/mosquitto`
3. 手动清理：`sudo ./script/cleanup_mosquitto_logs.sh`
4. 考虑完全禁用文件日志（见下方）

## 极端优化：完全禁用文件日志

如果磁盘空间非常紧张，可以完全禁用Mosquitto文件日志：

```bash
# 编辑日志配置文件
sudo nano /etc/mosquitto/conf.d/broker_tuner_logging.conf

# 添加以下内容：
log_dest none

# 重新加载Mosquitto
sudo systemctl reload mosquitto
```

**注意**：禁用日志后无法查看Mosquitto的运行日志，只能通过systemd日志查看：
```bash
sudo journalctl -u mosquitto -n 50
```

## 最佳实践

1. **训练前**：运行配置脚本设置日志级别和logrotate
2. **训练时**：启用`--cleanup-mosquitto-logs`参数
3. **训练后**：检查日志大小，必要时手动清理
4. **定期维护**：每周检查一次日志目录大小

## 预期效果

使用优化配置后：
- **日志文件大小**：从42GB降至几百MB
- **日志文件数量**：只保留3个最新文件
- **磁盘占用**：减少99%以上

## 相关文档

- [磁盘空间优化指南](./DISK_SPACE_OPTIMIZATION.md)
- [训练命令指南](./TRAINING_COMMAND.md)
