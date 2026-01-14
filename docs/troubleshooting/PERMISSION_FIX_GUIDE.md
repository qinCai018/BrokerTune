# CSV文件写入权限问题修复指南

## 问题描述

`action_throughput_log.csv` 文件没有被写入，通常是因为：

1. **训练进程以root用户运行**，但文件属于普通用户（qincai）
2. **文件权限不正确**，导致无法追加写入

## 当前状态

从检查结果看：
- 文件存在且有内容（6行，包括表头）
- 文件所有者：`qincai:qincai`
- 文件权限：`0644` (-rw-r--r--)
- 训练进程：以 `root` 用户运行

**问题**：root进程无法写入属于qincai的文件（即使权限是644，root可以写，但可能因为其他原因失败）

## 解决方案

### 方案1：修复文件权限（推荐）

```bash
# 确保文件可以被所有用户写入（临时方案）
sudo chmod 666 /home/qincai/userDir/BrokerTuner/checkpoints/action_throughput_log.csv

# 或者修改整个目录权限
sudo chmod -R 777 /home/qincai/userDir/BrokerTuner/checkpoints/
```

### 方案2：修改文件所有者（如果训练以root运行）

```bash
# 将文件所有者改为root（如果训练进程是root）
sudo chown root:root /home/qincai/userDir/BrokerTuner/checkpoints/action_throughput_log.csv

# 或者将整个目录改为root
sudo chown -R root:root /home/qincai/userDir/BrokerTuner/checkpoints/
```

### 方案3：不使用sudo运行训练（最佳方案）

**不要使用sudo运行训练脚本**：

```bash
# ❌ 错误：使用sudo
sudo ./script/run_train.sh --enable-workload ...

# ✅ 正确：不使用sudo
./script/run_train.sh --enable-workload ...
```

如果必须使用sudo（例如需要重启mosquitto），可以：
1. 配置sudoers允许无密码执行特定命令
2. 或者使用 `sudo -E` 保持环境变量

### 方案4：代码自动处理（已实现）

代码已经改进，会：
1. 检测文件权限问题
2. 打印详细的错误信息
3. 提供修复建议

## 验证修复

修复后，检查文件是否继续写入：

```bash
# 监控文件变化
watch -n 1 'wc -l /home/qincai/userDir/BrokerTuner/checkpoints/action_throughput_log.csv'

# 或者查看最后几行
tail -f /home/qincai/userDir/BrokerTuner/checkpoints/action_throughput_log.csv
```

## 预防措施

1. **统一用户**：确保训练脚本和文件使用相同的用户
2. **正确权限**：目录755，文件644（或666如果多用户需要）
3. **避免sudo**：除非必要，不要使用sudo运行训练脚本
