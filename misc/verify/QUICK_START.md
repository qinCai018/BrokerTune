# 吞吐量测试快速开始

## 快速运行

```bash
cd /home/qincai/userDir/BrokerTuner/verify

# 使用运行脚本（推荐）
./run_test.sh

# 或直接运行Python脚本
sudo python3 throughput_test.py
```

## 测试内容

- **2种Broker配置**：
  1. `max_inflight_messages=100`，其他默认
  2. 所有参数默认

- **12种工作负载组合**：
  - 消息大小：256B, 512B, 1024B
  - QoS：0, 1
  - 发布周期：10ms, 50ms
  - 发布端：100个
  - 接收端：10个

**总计**: 24个测试用例

## 输出结果

测试结果保存在 `throughput_test_results.csv`，包含：
- Broker配置
- 工作负载参数
- 吞吐量（msg/s）

## 注意事项

1. **需要sudo权限**（修改Broker配置）
2. **确保Mosquitto运行**：`systemctl status mosquitto`
3. **确保emqtt_bench可用**：会自动检测，或设置`EMQTT_BENCH_PATH`

## 测试时间

- 每个测试用例：约40-50秒
- 全部24个测试：约16-20分钟

## 详细文档

更多信息请参考: [README.md](README.md)
