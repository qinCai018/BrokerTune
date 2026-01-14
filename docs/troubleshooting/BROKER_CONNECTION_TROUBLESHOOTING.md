# Mosquitto Broker 连接问题排查指南

## 问题：Connection refused

当执行 `mosquitto_sub -t "test/topic"` 时出现 "Connection refused" 错误，通常表示：
1. Mosquitto Broker 服务未运行
2. Mosquitto Broker 配置不允许连接
3. 端口未监听
4. 配置文件有错误导致服务无法启动

## 快速检查步骤

### 1. 检查Mosquitto服务状态

```bash
# 检查服务状态
systemctl status mosquitto

# 如果未运行，启动服务
sudo systemctl start mosquitto

# 检查是否激活
systemctl is-active mosquitto
```

### 2. 检查端口监听

```bash
# 检查1883端口是否监听
sudo netstat -tlnp | grep 1883
# 或
sudo ss -tlnp | grep 1883
```

### 3. 检查Mosquitto进程

```bash
# 检查进程是否存在
ps aux | grep mosquitto | grep -v grep

# 检查PID
pgrep mosquitto
```

### 4. 检查配置文件

```bash
# 检查主配置文件
cat /etc/mosquitto/mosquitto.conf | grep -E "^listener|^port|^bind"

# 检查训练生成的配置文件
cat /etc/mosquitto/conf.d/broker_tuner.conf
```

### 5. 检查日志

```bash
# 查看系统日志
sudo journalctl -u mosquitto -n 50 --no-pager

# 或查看日志文件（如果配置了）
tail -50 /var/log/mosquitto/mosquitto.log
```

## 常见问题和解决方案

### 问题1：Mosquitto服务未运行

**症状**：
```bash
$ systemctl status mosquitto
● mosquitto.service - Mosquitto MQTT Broker
   Loaded: loaded (/lib/systemd/system/mosquitto.service; enabled; vendor preset: enabled)
   Active: inactive (dead)
```

**解决方案**：
```bash
# 启动服务
sudo systemctl start mosquitto

# 检查状态
systemctl status mosquitto
```

### 问题2：配置文件错误导致服务无法启动

**症状**：
```bash
$ sudo journalctl -u mosquitto -n 20
Error: Invalid configuration file
```

**解决方案**：
```bash
# 检查配置文件语法
sudo mosquitto -c /etc/mosquitto/mosquitto.conf -t

# 检查训练生成的配置
cat /etc/mosquitto/conf.d/broker_tuner.conf

# 如果有错误，可以临时删除训练配置
sudo rm /etc/mosquitto/conf.d/broker_tuner.conf
sudo systemctl restart mosquitto
```

### 问题3：端口未监听

**症状**：
```bash
$ sudo netstat -tlnp | grep 1883
(无输出)
```

**解决方案**：
```bash
# 检查配置文件中的监听配置
grep -E "^listener|^port" /etc/mosquitto/mosquitto.conf

# 确保有类似配置：
# listener 1883
# 或
# port 1883

# 如果没有，添加配置
sudo nano /etc/mosquitto/mosquitto.conf
# 添加：listener 1883

# 重启服务
sudo systemctl restart mosquitto
```

### 问题4：训练过程中Broker重启失败

**症状**：训练脚本执行后，Broker无法连接

**可能原因**：
- 训练生成的配置文件有错误
- Broker重启后配置未正确加载

**解决方案**：
```bash
# 1. 检查训练生成的配置文件
cat /etc/mosquitto/conf.d/broker_tuner.conf

# 2. 检查是否有语法错误
sudo mosquitto -c /etc/mosquitto/mosquitto.conf -t

# 3. 如果配置文件有问题，可以临时删除
sudo rm /etc/mosquitto/conf.d/broker_tuner.conf
sudo systemctl restart mosquitto

# 4. 重新运行训练
```

## 使用检查脚本

已创建检查脚本 `script/check_broker_connection.sh`：

```bash
cd /home/qincai/userDir/BrokerTuner
./script/check_broker_connection.sh
```

这个脚本会自动检查：
1. Mosquitto服务状态
2. Mosquitto进程
3. 端口监听
4. 连接测试
5. 配置文件
6. 日志信息

## 训练过程中的注意事项

### 1. Broker重启是正常的

训练过程中，Broker可能会被重启（当应用某些配置时）。这是正常的，但会导致：
- 所有现有连接断开
- 工作负载需要重新连接
- 短暂的服务不可用

### 2. 工作负载会自动恢复

训练脚本中的 `WorkloadHealthCheckCallback` 会每10步检查一次工作负载状态，如果停止会自动重启。

### 3. 手动订阅可能失败

如果在训练过程中手动订阅主题，可能会遇到连接问题，因为：
- Broker可能正在重启
- 工作负载可能正在重新连接
- 配置可能正在更新

## 验证Broker连接

### 方法1：使用mosquitto_sub测试

```bash
# 基本连接测试
mosquitto_sub -h 127.0.0.1 -p 1883 -t "test/topic" -C 1

# 持续订阅
mosquitto_sub -h 127.0.0.1 -p 1883 -t "test/topic" -v
```

### 方法2：使用mosquitto_pub测试

```bash
# 发布一条消息
mosquitto_pub -h 127.0.0.1 -p 1883 -t "test/topic" -m "Hello"

# 在另一个终端订阅
mosquitto_sub -h 127.0.0.1 -p 1883 -t "test/topic"
```

### 方法3：检查$SYS主题

```bash
# 订阅系统主题（Broker状态）
mosquitto_sub -h 127.0.0.1 -p 1883 -t "\$SYS/#" -v
```

## 如果问题仍然存在

1. **停止训练**：如果训练正在运行，先停止训练
2. **检查配置**：运行检查脚本 `./script/check_broker_connection.sh`
3. **查看日志**：`sudo journalctl -u mosquitto -n 100`
4. **重置配置**：如果训练配置有问题，删除 `/etc/mosquitto/conf.d/broker_tuner.conf`
5. **重启Broker**：`sudo systemctl restart mosquitto`
6. **验证连接**：`mosquitto_sub -h 127.0.0.1 -p 1883 -t "test/topic" -C 1`

## 联系支持

如果问题仍然无法解决，请提供：
1. `systemctl status mosquitto` 的输出
2. `sudo journalctl -u mosquitto -n 50` 的输出
3. `/etc/mosquitto/conf.d/broker_tuner.conf` 的内容
4. 训练脚本的输出
