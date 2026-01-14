#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BrokerTuner - 性能指标收集脚本
启动 emqtt_bench 负载并收集 Broker 性能指标

使用示例:
    python collect_metrics.py --broker-host 127.0.0.1 --broker-port 1883
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from script.workload import WorkloadManager, WorkloadConfig
from environment.utils import MQTTSampler, read_proc_metrics
from environment.config import MQTTConfig, ProcConfig


def collect_broker_metrics(
    broker_host: str = "127.0.0.1",
    broker_port: int = 1883,
    emqtt_bench_path: str = None,
    output_file: str = None,
) -> Dict[str, Any]:
    """
    启动负载并收集 Broker 性能指标
    
    Args:
        broker_host: Broker 地址
        broker_port: Broker 端口
        emqtt_bench_path: emqtt_bench 可执行文件路径
        output_file: 输出文件路径（JSON格式）
    
    Returns:
        包含性能指标的字典
    """
    print("=" * 60)
    print("BrokerTuner - 性能指标收集")
    print("=" * 60)
    print(f"Broker 地址: {broker_host}:{broker_port}")
    print()
    
    # 配置工作负载
    workload_config = WorkloadConfig(
        num_publishers=100,          # 100个发布者
        num_subscribers=10,          # 10个订阅者
        topic="test/topic",
        message_size=1024,           # 每条消息1024字节
        qos=1,                      # QoS 1
        publisher_interval_ms=10,   # 每个发布者每10毫秒发布一条消息
        duration=0,                 # 持续运行直到手动停止
    )
    
    print("工作负载配置:")
    print(f"  - 发布者数量: {workload_config.num_publishers}")
    print(f"  - 订阅者数量: {workload_config.num_subscribers}")
    print(f"  - 消息大小: {workload_config.message_size} 字节")
    print(f"  - QoS: {workload_config.qos}")
    print(f"  - 发布间隔: {workload_config.publisher_interval_ms} 毫秒")
    print()
    
    # 创建工作负载管理器
    workload = None
    sampler = None
    
    try:
        print("1. 启动工作负载...")
        workload = WorkloadManager(
            broker_host=broker_host,
            broker_port=broker_port,
            emqtt_bench_path=emqtt_bench_path,
        )
        
        workload.start(config=workload_config)
        print("✅ 工作负载已启动")
        print()
        
        # 等待负载稳定（50秒）
        print("2. 等待负载稳定（50秒）...")
        for i in range(50, 0, -5):
            print(f"   剩余时间: {i} 秒...", end='\r')
            time.sleep(5)
        print("   负载已稳定" + " " * 20)  # 清除进度行
        print()
        
        # 初始化 MQTT 采样器
        print("3. 初始化性能指标采样器...")
        mqtt_config = MQTTConfig(
            host=broker_host,
            port=broker_port,
            timeout_sec=12.0,
        )
        sampler = MQTTSampler(mqtt_config)
        print("✅ 采样器已初始化")
        print()
        
        # 初始化进程指标配置
        proc_config = ProcConfig()
        print(f"   检测到 Mosquitto PID: {proc_config.pid}")
        print()
        
        # 收集 Broker 指标
        print("4. 收集 Broker 性能指标...")
        broker_metrics = sampler.sample(timeout_sec=12.0)
        print(f"✅ 已收集 {len(broker_metrics)} 条 Broker 指标")
        print()
        
        # 收集进程指标
        print("5. 收集进程性能指标...")
        cpu_ratio, mem_ratio, ctxt_ratio = read_proc_metrics(proc_config)
        print(f"✅ CPU 使用率: {cpu_ratio:.4f}")
        print(f"✅ 内存使用率: {mem_ratio:.4f}")
        print(f"✅ 上下文切换率: {ctxt_ratio:.4f}")
        print()
        
        # 构建结果字典
        result = {
            "timestamp": datetime.now().isoformat(),
            "broker_host": broker_host,
            "broker_port": broker_port,
            "workload_config": {
                "num_publishers": workload_config.num_publishers,
                "num_subscribers": workload_config.num_subscribers,
                "message_size": workload_config.message_size,
                "qos": workload_config.qos,
                "publisher_interval_ms": workload_config.publisher_interval_ms,
            },
            "broker_metrics": broker_metrics,
            "process_metrics": {
                "pid": proc_config.pid,
                "cpu_ratio": float(cpu_ratio),
                "mem_ratio": float(mem_ratio),
                "ctxt_ratio": float(ctxt_ratio),
            },
        }
        
        # 保存到文件
        if output_file:
            print(f"6. 保存结果到文件: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print("✅ 结果已保存")
            print()
        else:
            # 如果没有指定输出文件，使用默认文件名
            default_output = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            print(f"6. 保存结果到文件: {default_output}")
            with open(default_output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print("✅ 结果已保存")
            print()
        
        # 打印关键指标摘要
        print("=" * 60)
        print("性能指标摘要")
        print("=" * 60)
        
        # 连接数
        clients_connected = broker_metrics.get("$SYS/broker/clients/connected", 0)
        print(f"连接数: {clients_connected}")
        
        # 消息速率
        messages_rate_1min = broker_metrics.get("$SYS/broker/load/messages/received/1min", 0)
        messages_received_total = broker_metrics.get("$SYS/broker/messages/received", 0)
        print(f"消息接收速率 (1分钟): {messages_rate_1min} msg/s")
        print(f"消息接收总数: {messages_received_total}")
        
        # 消息发送
        messages_sent_total = broker_metrics.get("$SYS/broker/messages/sent", 0)
        print(f"消息发送总数: {messages_sent_total}")
        
        # CPU/内存
        print(f"CPU 使用率: {cpu_ratio:.2%}")
        print(f"内存使用率: {mem_ratio:.2%}")
        print(f"上下文切换率: {ctxt_ratio:.4f}")
        
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        # 清理资源
        print("\n7. 清理资源...")
        if sampler:
            sampler.close()
            print("✅ 采样器已关闭")
        
        if workload:
            workload.stop()
            print("✅ 工作负载已停止")
        
        print("\n完成")


def main():
    parser = argparse.ArgumentParser(
        description="启动 emqtt_bench 负载并收集 Broker 性能指标"
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
        help="emqtt_bench 可执行文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（JSON格式，默认：metrics_YYYYMMDD_HHMMSS.json）",
    )
    
    args = parser.parse_args()
    
    try:
        collect_broker_metrics(
            broker_host=args.broker_host,
            broker_port=args.broker_port,
            emqtt_bench_path=args.emqtt_bench_path,
            output_file=args.output,
        )
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n程序异常退出: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
