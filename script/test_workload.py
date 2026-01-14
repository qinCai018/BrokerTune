#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试工作负载脚本

用于验证 emqtt_bench 工作负载是否能正常运行。
测试指定的工作负载配置（100个发布者、10个订阅者、15ms间隔、512B消息、QoS=1）

使用方法:
    python3 script/test_workload.py
    # 或
    python3 script/test_workload.py --duration 30  # 运行30秒
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.workload import WorkloadManager, WorkloadConfig


def test_workload(
    duration: int = 60,
    broker_host: str = "127.0.0.1",
    broker_port: int = 1883,
    emqtt_bench_path: str | None = None,
) -> bool:
    """
    测试工作负载配置
    
    Args:
        duration: 测试持续时间（秒）
        broker_host: MQTT Broker 地址
        broker_port: MQTT Broker 端口
        emqtt_bench_path: emqtt_bench 可执行文件路径
        
    Returns:
        True 如果测试成功，False 如果失败
    """
    print("=" * 80)
    print("工作负载测试脚本")
    print("=" * 80)
    print(f"测试配置:")
    print(f"  - 发布者数量: 100")
    print(f"  - 订阅者数量: 10")
    print(f"  - 发布者间隔: 15ms")
    print(f"  - 消息大小: 512B")
    print(f"  - QoS: 1")
    print(f"  - 测试时长: {duration} 秒")
    print(f"  - Broker: {broker_host}:{broker_port}")
    print("=" * 80)
    print()
    
    # 创建工作负载配置
    workload_config = WorkloadConfig(
        num_publishers=100,
        num_subscribers=10,
        topic="test/topic",
        message_size=512,
        qos=1,
        publisher_interval_ms=15,
        duration=0,  # 持续运行，手动停止
    )
    
    # 计算预期消息速率
    messages_per_publisher_per_sec = 1000.0 / workload_config.publisher_interval_ms
    total_message_rate = int(messages_per_publisher_per_sec * workload_config.num_publishers)
    
    print(f"预期消息速率:")
    print(f"  - 每个发布者: ~{messages_per_publisher_per_sec:.2f} msg/s")
    print(f"  - 总计: ~{total_message_rate} msg/s")
    print()
    
    # 创建工作负载管理器
    try:
        workload = WorkloadManager(
            broker_host=broker_host,
            broker_port=broker_port,
            emqtt_bench_path=emqtt_bench_path,
        )
    except Exception as e:
        print(f"❌ 创建工作负载管理器失败: {e}")
        print("\n提示:")
        print("1. 确保已安装 emqtt_bench:")
        print("   git clone https://github.com/emqx/emqtt-bench.git")
        print("   cd emqtt-bench && make")
        print("2. 或者设置 EMQTT_BENCH_PATH 环境变量")
        print("3. 或者使用 --emqtt-bench-path 参数指定路径")
        return False
    
    # 启动工作负载
    print("启动工作负载...")
    try:
        workload.start(config=workload_config)
        print("✅ 工作负载启动成功！")
        print()
    except Exception as e:
        print(f"❌ 工作负载启动失败: {e}")
        return False
    
    # 检查进程状态
    print("检查工作负载进程状态...")
    time.sleep(2)  # 等待进程稳定
    
    if not workload.is_running():
        print("❌ 工作负载进程已退出")
        return False
    
    print("✅ 工作负载进程运行正常")
    print()
    
    # 运行测试
    print(f"运行测试 {duration} 秒...")
    print("提示: 可以使用以下命令监听消息:")
    print(f"  mosquitto_sub -h {broker_host} -t '{workload_config.topic}' -v")
    print()
    
    start_time = time.time()
    elapsed = 0
    
    try:
        while elapsed < duration:
            time.sleep(1)
            elapsed = int(time.time() - start_time)
            remaining = duration - elapsed
            
            # 检查进程是否还在运行
            if not workload.is_running():
                print(f"\n❌ 工作负载进程在运行 {elapsed} 秒后退出")
                return False
            
            # 显示进度
            if elapsed % 10 == 0 or remaining <= 5:
                print(f"  运行中... {elapsed}/{duration} 秒 (剩余 {remaining} 秒)")
        
        print(f"\n✅ 测试完成！工作负载正常运行了 {duration} 秒")
        print()
        
    except KeyboardInterrupt:
        print(f"\n⚠️  测试被用户中断（已运行 {elapsed} 秒）")
        print()
    
    # 停止工作负载
    print("停止工作负载...")
    try:
        workload.stop()
        print("✅ 工作负载已停止")
        return True
    except Exception as e:
        print(f"⚠️  停止工作负载时出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="测试 emqtt_bench 工作负载配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行60秒测试（默认）
  python3 script/test_workload.py
  
  # 运行30秒测试
  python3 script/test_workload.py --duration 30
  
  # 指定 emqtt_bench 路径
  python3 script/test_workload.py --emqtt-bench-path /path/to/emqtt_bench
  
  # 指定不同的 broker
  python3 script/test_workload.py --broker-host 192.168.1.100 --broker-port 1883
        """
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="测试持续时间（秒，默认：60）",
    )
    
    parser.add_argument(
        "--broker-host",
        type=str,
        default="127.0.0.1",
        help="MQTT Broker 地址（默认：127.0.0.1）",
    )
    
    parser.add_argument(
        "--broker-port",
        type=int,
        default=1883,
        help="MQTT Broker 端口（默认：1883）",
    )
    
    parser.add_argument(
        "--emqtt-bench-path",
        type=str,
        default=None,
        help="emqtt_bench 可执行文件路径（默认：从环境变量或 PATH 查找）",
    )
    
    args = parser.parse_args()
    
    # 运行测试
    success = test_workload(
        duration=args.duration,
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        emqtt_bench_path=args.emqtt_bench_path,
    )
    
    # 退出
    if success:
        print("=" * 80)
        print("✅ 测试通过！工作负载配置正常，可以用于训练")
        print("=" * 80)
        sys.exit(0)
    else:
        print("=" * 80)
        print("❌ 测试失败！请检查配置和 emqtt_bench 安装")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
