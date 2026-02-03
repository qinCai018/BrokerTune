#!/usr/bin/env python3
"""
一次性验证 $SYS 指标与吞吐估算是否一致。

输出：
- $SYS/broker/load/messages/received/1min（原始值，通常为每分钟）
- $SYS/broker/load/messages/received/1min_per_sec（换算为 msg/s）
- $SYS/broker/messages/received_rate（采样窗口估算）
- $SYS/broker/messages/received（累计）
- 估算使用的速率与归一化吞吐
可选：
- 自动启动工作负载（emqtt_bench），等待稳定后采样
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment import EnvConfig
from environment.utils import MQTTSampler, _parse_numeric_payload

try:
    from script.workload import WorkloadManager
except ImportError:
    WorkloadManager = None  # type: ignore


def _choose_rate(rate_1min_per_sec: Optional[float], rate_derived: Optional[float]) -> Tuple[float, str]:
    if rate_1min_per_sec is not None and rate_1min_per_sec > 0:
        return float(rate_1min_per_sec), "1min_per_sec"
    if rate_derived is not None:
        return float(rate_derived), "sample_window_rate"
    return 0.0, "missing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check $SYS metrics and throughput estimation")
    parser.add_argument("--samples", type=int, default=3, help="采样次数（默认：3）")
    parser.add_argument("--timeout", type=float, default=None, help="单次采样超时（秒）")
    parser.add_argument("--sleep", type=float, default=3.0, help="两次采样间隔（秒）")
    parser.add_argument("--host", type=str, default=None, help="Broker host（默认使用 EnvConfig）")
    parser.add_argument("--port", type=int, default=None, help="Broker port（默认使用 EnvConfig）")
    parser.add_argument(
        "--start-workload",
        action="store_true",
        help="采样前启动工作负载（emqtt_bench）",
    )
    parser.add_argument("--workload-publishers", type=int, default=100, help="发布者数量")
    parser.add_argument("--workload-subscribers", type=int, default=10, help="订阅者数量")
    parser.add_argument("--workload-topic", type=str, default="test/topic", help="MQTT 主题")
    parser.add_argument(
        "--workload-message-rate",
        type=int,
        default=None,
        help="总消息速率（msg/s），默认按发布间隔估算",
    )
    parser.add_argument(
        "--workload-publisher-interval-ms",
        type=int,
        default=15,
        help="发布者间隔（ms），默认 15ms",
    )
    parser.add_argument("--workload-message-size", type=int, default=512, help="消息大小（字节）")
    parser.add_argument("--workload-qos", type=int, default=1, choices=[0, 1, 2], help="QoS")
    parser.add_argument("--workload-duration", type=int, default=0, help="工作负载持续时间（秒，0表示持续）")
    parser.add_argument("--workload-warmup-sec", type=float, default=12.0, help="工作负载启动后等待稳定时间（秒）")
    parser.add_argument("--emqtt-bench-path", type=str, default=None, help="emqtt_bench 可执行文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EnvConfig()
    if args.host:
        cfg.mqtt.host = args.host
    if args.port:
        cfg.mqtt.port = args.port
    if args.timeout is not None:
        cfg.mqtt.timeout_sec = args.timeout

    workload = None
    workload_started = False
    if args.start_workload:
        if WorkloadManager is None:
            raise RuntimeError("无法导入 WorkloadManager，请确认 script/workload.py 存在")
        workload = WorkloadManager(
            broker_host=cfg.mqtt.host,
            broker_port=cfg.mqtt.port,
            emqtt_bench_path=args.emqtt_bench_path,
        )
        if args.workload_message_rate is None:
            per_publisher = 1000.0 / args.workload_publisher_interval_ms
            total_rate = int(per_publisher * args.workload_publishers)
        else:
            total_rate = args.workload_message_rate
        print("[workload] 启动工作负载...")
        print(f"[workload] publishers={args.workload_publishers}, subscribers={args.workload_subscribers}")
        print(f"[workload] topic={args.workload_topic}, qos={args.workload_qos}")
        print(f"[workload] message_size={args.workload_message_size}B, total_rate={total_rate} msg/s")
        workload.start(
            num_publishers=args.workload_publishers,
            num_subscribers=args.workload_subscribers,
            topic=args.workload_topic,
            message_rate=total_rate,
            message_size=args.workload_message_size,
            duration=args.workload_duration,
            qos=args.workload_qos,
        )
        workload_started = True
        if args.workload_warmup_sec > 0:
            print(f"[workload] 等待稳定 {args.workload_warmup_sec:.1f} 秒...")
            time.sleep(args.workload_warmup_sec)

    sampler = MQTTSampler(cfg.mqtt)
    try:
        for i in range(args.samples):
            start = time.time()
            metrics = sampler.sample(timeout_sec=cfg.mqtt.timeout_sec)
            elapsed = time.time() - start

            rate_1min = metrics.get("$SYS/broker/load/messages/received/1min")
            rate_1min_per_sec = metrics.get("$SYS/broker/load/messages/received/1min_per_sec")
            rate_derived = metrics.get("$SYS/broker/messages/received_rate")
            total_received = metrics.get("$SYS/broker/messages/received")
            clients = metrics.get("$SYS/broker/clients/connected")
            uptime_sec = metrics.get("$SYS/broker/uptime")
            uptime_raw = None
            if uptime_sec is None:
                uptime_raw = metrics.get("$SYS/broker/uptime_raw")
                if uptime_raw is not None:
                    uptime_sec = _parse_numeric_payload(str(uptime_raw))

            if rate_1min_per_sec is None and rate_1min is not None:
                divisor = cfg.mqtt.rate_1min_divisor
                if divisor and divisor > 0:
                    rate_1min_per_sec = rate_1min / divisor

            chosen_rate, source = _choose_rate(rate_1min_per_sec, rate_derived)
            throughput_norm = chosen_rate / 10000.0

            print("=" * 80)
            print(f"Sample {i + 1}/{args.samples} | elapsed={elapsed:.2f}s")
            print(f"  clients_connected: {clients}")
            print(f"  broker_uptime:     {uptime_sec}")
            if uptime_raw is not None:
                print(f"  uptime_raw:        {uptime_raw}")
            print(f"  received_1min:     {rate_1min}")
            print(f"  1min_per_sec:      {rate_1min_per_sec}")
            print(f"  received_rate:     {rate_derived}")
            print(f"  received_total:    {total_received}")
            print(f"  chosen_rate:       {chosen_rate:.2f} ({source})")
            print(f"  throughput_norm:   {throughput_norm:.6f}")

            if i < args.samples - 1 and args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        sampler.close()
        if workload is not None and workload_started:
            try:
                workload.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
