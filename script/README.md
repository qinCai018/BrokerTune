# 执行测试命令

```python
export MOSQUITTO_PID=$(pgrep mosquitto)

python3 script/test_mosquitto.py \
            --show \                              # 显示详细过程: 显示每步的 state、action、reward
            --device cpu \                        # 使用 CPU 设备
            --deterministic \                     # 使用确定性策略
            --enable-workload \                   # 启用工作负载
            --workload-publishers 100 \           # 发布者数量100
            --workload-subscribers 100 \          # 订阅者数量100
            --workload-topic "test/topic" \       # MQTT 主题
            --workload-message-rate 100 \         # 每秒消息数,所有发布者合计的每秒消息数（msg/s）
            --workload-message-size 100           # 消息大小（字节）
```

# 函数调用栈
```c
1. 命令行执行
   └─ python3 script/test_mosquitto.py --show
      │
      ├─ 文件: script/test_mosquitto.py
      └─ 入口: if __name__ == "__main__": (第 151 行)
         │
         ├─ 2. 解析参数
         │  └─ args = parser.parse_args() (第 225 行)
         │
         ├─ 3. 设置环境变量
         │  └─ os.environ["BROKER_TUNER_DRY_RUN"] = "true" (第 229 行)
         │
         ├─ 4. 创建环境
         │  └─ env = make_env() (第 231 行)
         │     │
         │     └─ 调用: tuner/utils.py::make_env() (第 20 行)
         │        └─ 返回: MosquittoBrokerEnv 实例
         │
         ├─ 5. 创建/加载模型
         │  └─ model = make_ddpg_model(env, device=args.device) (第 239 行)
         │     │
         │     └─ 调用: tuner/utils.py::make_ddpg_model() (第 29 行)
         │        └─ 返回: DDPG 模型实例
         │
         └─ 6. 调用 play() 函数
            └─ play(model, env, show=args.show, ...) (第 259 行)
               │
               └─ 文件: script/test_mosquitto.py
                  └─ 函数: play() (第 54 行)
                     │
                     ├─ 7. 环境重置
                     │  └─ s = env.reset() (第 86 行)
                     │     │
                     │     └─ 调用: environment/broker.py::MosquittoBrokerEnv.reset() (第 60 行)
                     │
                     └─ 8. 主循环 (while not done:)
                        │
                        ├─ 9. 模型预测动作
                        │  └─ a, _ = model.predict(s, deterministic=deterministic) (第 102 行)
                        │     │
                        │     └─ 调用: stable_baselines3::DDPG.predict()
                        │
                        └─ 10. 执行动作
                           └─ ns, r, done, info = env.step(a) (第 105 行)
                              │
                              └─ 文件: environment/broker.py
                                 └─ 函数: MosquittoBrokerEnv.step() (第 71 行)
                                    │
                                    ├─ 11. 解码动作
                                    │  └─ knobs = self.knob_space.decode_action(action) (第 83 行)
                                    │     │
                                    │     └─ 调用: environment/knobs.py::BrokerKnobSpace.decode_action() (第 71 行)
                                    │        └─ 返回: Dict[str, Any] (配置字典)
                                    │
                                    └─ 12. 应用配置 ⭐
                                       └─ apply_knobs(knobs) (第 84 行)
                                          │
                                          └─ 文件: environment/knobs.py
                                             └─ 函数: apply_knobs() (第 146 行)
                                                │
                                                ├─ 检查 dry_run 模式 (第 167-168 行)
                                                ├─ 构建配置行 (第 175-215 行)
                                                └─ 打印配置 (第 218-221 行) ← 你看到的输出在这里
```


# 验证环境


