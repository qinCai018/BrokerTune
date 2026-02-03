# 训练流程说明（函数与模块调用链）

本文描述 Mosquitto 调优系统的训练流程，以及训练过程中调用的关键模块与函数。路径均基于仓库根目录 `BrokerTuner/`。

**入口与参数解析**
1. 训练入口：`tuner/train.py` 中的 `main()`
1. 命令行参数解析：`tuner/train.py` 中的 `parse_args()`
1. 环境配置实例：`environment/config.py` 中的 `EnvConfig`

**工作负载准备**
1. 创建工作负载管理器：`script/workload.py` 中的 `WorkloadManager.__init__()`
1. 构造工作负载配置：`script/workload.py` 中的 `WorkloadConfig`
1. 启动工作负载：`script/workload.py` 中的 `WorkloadManager.start()`

**环境创建与包装**
1. 创建环境：`tuner/utils.py` 中的 `make_env()`，内部构造 `environment/broker.py` 中的 `MosquittoBrokerEnv`
1. 训练日志包装：`tuner/train.py` 中的 `ActionThroughputLoggerWrapper`
1. 统计包装：`stable_baselines3.common.monitor.Monitor`

**训练开始前的默认基线**
1. 记录默认性能：`tuner/train.py` 中的 `record_default_baseline()`
1. 触发环境重置：`environment/broker.py` 中的 `MosquittoBrokerEnv.reset()`
1. 应用默认配置并重启 Broker：`environment/knobs.py` 中的 `apply_knobs()`
1. 写入配置文件并启动：`apply_knobs()` 内部执行 `mosquitto -c broker_tuner.conf -d`

**每一步训练交互**
1. 强化学习调用：`stable_baselines3.DDPG.learn()` 进入训练循环
1. 环境单步交互：`environment/broker.py` 中的 `MosquittoBrokerEnv.step()`
1. 动作解码为配置：`environment/knobs.py` 中的 `BrokerKnobSpace.decode_action()`
1. 应用配置并重启 Broker：`environment/knobs.py` 中的 `apply_knobs()`
1. 指标采样：`environment/utils.py` 中的 `MQTTSampler.sample()`
1. 进程指标采样：`environment/utils.py` 中的 `read_proc_metrics()`
1. 状态向量构建：`environment/utils.py` 中的 `build_state_vector()`
1. 奖励计算：`environment/broker.py` 中的 `MosquittoBrokerEnv._compute_reward()`

**日志与回调**
1. 训练日志：`tuner/train.py` 中的 `ActionThroughputLoggerWrapper.step()` 写入 `action_throughput_log.csv`
1. 进度与Checkpoint：`tuner/train.py` 中的 `ProgressBarCallback`、`CheckpointCallback`
1. 工作负载健康检查：`tuner/train.py` 中的 `WorkloadHealthCheckCallback`

**训练结束**
1. 保存模型：`tuner/utils.py` 中的 `save_model()`
1. 关闭环境：`tuner/train.py` 中的 `env.close()`

**关键输出文件**
1. 训练过程日志：`checkpoints/action_throughput_log.csv`
1. 默认基线：`checkpoints/baseline_metrics.json`
1. 训练统计：`checkpoints/logs/progress.csv`
1. 最终模型：`checkpoints/ddpg_mosquitto_final.zip`
