# -*- coding: utf-8 -*-
"""
工作负载emqtt_bench 的使用示例

演示如何使用 WorkloadManager 和测试函数进行 Mosquitto Broker 测试。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from script.workload import WorkloadManager, WorkloadConfig
from script.test_mosquitto import play
from tuner.utils import make_env, make_ddpg_model, load_model


def example_1_basic_usage():
    """示例1: 基本使用 - 手动管理工作负载"""
    print("=" * 50)
    print("示例1: 基本使用")
    print("=" * 50)
    
    # 创建工作负载管理器
    workload = WorkloadManager(
        broker_host="127.0.0.1",
        broker_port=1883,
    )
    
    try:
        # 启动工作负载
        workload.start(
            num_publishers=100,
            num_subscribers=100,
            topic="test/topic",
            message_rate=100,  # 每秒100条消息
            message_size=100,  # 每条消息100字节
        )
        
        print("工作负载已启动，可以进行测试...")
        # 在这里进行你的测试
        # ...
        
    finally:
        # 停止工作负载
        workload.stop()


def example_2_context_manager():
    """示例2: 使用上下文管理器自动管理"""
    print("=" * 50)
    print("示例2: 使用上下文管理器")
    print("=" * 50)
    
    with WorkloadManager() as workload:
        workload.start(
            num_publishers=50,
            num_subscribers=50,
            topic="test/topic",
            message_rate=50,
        )
        # 工作负载会自动在退出时停止
        print("工作负载运行中...")


def example_3_with_test():
    """示例3: 结合测试函数使用（使用未训练的模型观察训练过程）"""
    print("=" * 50)
    print("示例3: 结合测试函数使用")
    print("=" * 50)
    print("注意: 此示例使用未训练的模型，目的是观察强化学习训练过程")
    print("      查看 state、action、reward 的输出")
    print("=" * 50)
    
    # 创建环境和未训练的模型（用于测试观察）
    env = make_env()
    model = make_ddpg_model(env, device="cpu")
    
    # 创建工作负载
    workload = WorkloadManager(
        broker_host=env.cfg.mqtt.host,
        broker_port=env.cfg.mqtt.port,
    )
    
    try:
        # 启动工作负载
        workload.start(
            num_publishers=100,
            num_subscribers=100,
            topic="test/topic",
            message_rate=100,
        )
        
        # 运行测试（show=True 会显示详细的 state、action、reward）
        state, action, reward, reward_sum = play(
            model, env, 
            workload=workload,
            show=True,  # 显示详细的训练过程
        )
        
        print(f"\n测试完成，总奖励: {reward_sum:.6f}")
        
    finally:
        workload.stop()
        env.close()


def example_4_custom_config():
    """示例4: 使用自定义配置"""
    print("=" * 50)
    print("示例4: 使用自定义配置")
    print("=" * 50)
    
    config = WorkloadConfig(
        num_publishers=200,
        num_subscribers=200,
        topic="custom/topic",
        message_size=500,
        qos=1,  # 使用QoS 1
        publisher_interval_ms=50,
        duration=300,  # 运行300秒后自动停止
    )
    
    workload = WorkloadManager()
    try:
        workload.start(config=config)
        print("自定义工作负载已启动...")
        # ...
    finally:
        workload.stop()


def example_5_connection_test():
    """示例5: 仅连接测试（不发布/订阅消息）"""
    print("=" * 50)
    print("示例5: 连接测试")
    print("=" * 50)
    
    config = WorkloadConfig(
        num_connections=1000,  # 1000个连接
        num_publishers=0,
        num_subscribers=0,
    )
    
    workload = WorkloadManager()
    try:
        workload.start(config=config)
        print("连接测试已启动...")
        # ...
    finally:
        workload.stop()


if __name__ == "__main__":
    print("工作负载使用示例")
    print("\n注意: 这些示例需要:")
    print("1. Mosquitto Broker 正在运行")
    print("2. emqtt_bench 已安装并在 PATH 中，或设置 EMQTT_BENCH_PATH 环境变量")
    print("3. 示例3使用未训练的模型，用于观察强化学习训练过程和查看 state、action、reward")
    print("\n取消注释下面的行来运行示例:\n")
    
    # example_1_basic_usage()
    # example_2_context_manager()
    # example_3_with_test()
    # example_4_custom_config()
    # example_5_connection_test()
