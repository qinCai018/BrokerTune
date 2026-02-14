"""
模型评估入口：
- 加载已训练好的 DDPG 模型
- 对比 baseline（默认配置）与 RL 策略的吞吐/时延/奖励
"""

from __future__ import annotations

import argparse
import json
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from environment import EnvConfig
from .utils import load_model, make_env

try:
    from script.workload import WorkloadConfig, WorkloadManager

    WORKLOAD_AVAILABLE = True
except Exception:
    WORKLOAD_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained DDPG model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型文件路径（.zip）",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="评估 episode 数量",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="评估设备，例如 'cpu' 或 'cuda'（默认：cpu）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认：42）",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        default=False,
        help="跳过 baseline 对比，仅评估 RL 策略",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="可选：将评估结果写入 JSON 文件",
    )
    parser.add_argument(
        "--enable-workload",
        action="store_true",
        help="评估期间启用工作负载（推荐，结果更可信）",
    )
    parser.add_argument(
        "--workload-publishers",
        type=int,
        default=100,
        help="工作负载发布者数量（默认：100）",
    )
    parser.add_argument(
        "--workload-subscribers",
        type=int,
        default=10,
        help="工作负载订阅者数量（默认：10）",
    )
    parser.add_argument(
        "--workload-topic",
        type=str,
        default="test/topic",
        help="工作负载 MQTT 主题（默认：test/topic）",
    )
    parser.add_argument(
        "--workload-publisher-interval-ms",
        type=int,
        default=15,
        help="发布者发送间隔（毫秒，默认：15）",
    )
    parser.add_argument(
        "--workload-message-size",
        type=int,
        default=512,
        help="消息大小（字节，默认：512）",
    )
    parser.add_argument(
        "--workload-qos",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="QoS级别（默认：1）",
    )
    parser.add_argument(
        "--emqtt-bench-path",
        type=str,
        default=None,
        help="emqtt_bench 可执行文件路径（可选）",
    )
    return parser.parse_args()


def _extract_metrics(obs: np.ndarray, info: Dict[str, Any]) -> Dict[str, float]:
    throughput_norm = float(info.get("throughput_norm", obs[1] if len(obs) > 1 else 0.0))
    throughput_msg_per_sec = float(
        info.get("throughput_msg_per_sec", throughput_norm * 10000.0)
    )
    latency_p50_ms = float(info.get("latency_p50_ms", (obs[5] * 100.0) if len(obs) > 5 else 0.0))
    latency_p95_ms = float(info.get("latency_p95_ms", (obs[6] * 100.0) if len(obs) > 6 else 0.0))
    return {
        "throughput_norm": throughput_norm,
        "throughput_msg_per_sec": throughput_msg_per_sec,
        "latency_p50_ms": latency_p50_ms,
        "latency_p95_ms": latency_p95_ms,
    }


def run_episode(
    env,
    action_fn: Callable[[np.ndarray], np.ndarray],
    seed: Optional[int] = None,
) -> Dict[str, float]:
    if seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)

    done = False
    total_reward = 0.0
    throughput_values: List[float] = []
    latency_p50_values: List[float] = []
    latency_p95_values: List[float] = []

    while not done:
        action = action_fn(obs)
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)

        total_reward += float(reward)
        metrics = _extract_metrics(obs, info)
        throughput_values.append(metrics["throughput_msg_per_sec"])
        latency_p50_values.append(metrics["latency_p50_ms"])
        latency_p95_values.append(metrics["latency_p95_ms"])

    return {
        "episode_reward": total_reward,
        "episode_throughput_msg_per_sec": float(mean(throughput_values)) if throughput_values else 0.0,
        "episode_latency_p50_ms": float(mean(latency_p50_values)) if latency_p50_values else 0.0,
        "episode_latency_p95_ms": float(mean(latency_p95_values)) if latency_p95_values else 0.0,
    }


def summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    rewards = [r["episode_reward"] for r in results]
    tps = [r["episode_throughput_msg_per_sec"] for r in results]
    p50s = [r["episode_latency_p50_ms"] for r in results]
    p95s = [r["episode_latency_p95_ms"] for r in results]
    return {
        "mean_reward": float(mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_throughput_msg_per_sec": float(mean(tps)) if tps else 0.0,
        "mean_latency_p50_ms": float(mean(p50s)) if p50s else 0.0,
        "mean_latency_p95_ms": float(mean(p95s)) if p95s else 0.0,
    }


def _run_policy_eval(
    env,
    n_episodes: int,
    seed: int,
    label: str,
    action_fn: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, float]:
    per_episode: List[Dict[str, float]] = []
    for i in range(n_episodes):
        episode_stats = run_episode(env, action_fn=action_fn, seed=seed + i)
        per_episode.append(episode_stats)
        print(
            f"[{label}] Episode {i + 1}/{n_episodes} | "
            f"reward={episode_stats['episode_reward']:.4f}, "
            f"throughput={episode_stats['episode_throughput_msg_per_sec']:.2f} msg/s, "
            f"p50={episode_stats['episode_latency_p50_ms']:.2f} ms, "
            f"p95={episode_stats['episode_latency_p95_ms']:.2f} ms"
        )
    return summarize(per_episode)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    workload = None
    if args.enable_workload:
        if not WORKLOAD_AVAILABLE:
            raise RuntimeError("未找到 script.workload.WorkloadManager，无法启用 --enable-workload")
        workload = WorkloadManager(
            broker_host="127.0.0.1",
            broker_port=1883,
            emqtt_bench_path=args.emqtt_bench_path,
        )
        workload_cfg = WorkloadConfig(
            num_publishers=args.workload_publishers,
            num_subscribers=args.workload_subscribers,
            topic=args.workload_topic,
            message_size=args.workload_message_size,
            qos=args.workload_qos,
            publisher_interval_ms=args.workload_publisher_interval_ms,
            duration=0,
        )
        workload.start(config=workload_cfg)
        print("[评估] 工作负载已启动，等待稳定（10秒）...")
        import time

        time.sleep(10.0)

    env_cfg = EnvConfig()
    env = make_env(env_cfg, workload_manager=workload)

    try:
        model = load_model(args.model_path, env, device=args.device)

        default_action = env.knob_space.get_default_action().astype(np.float32)
        baseline_summary = None
        if not args.skip_baseline:
            baseline_summary = _run_policy_eval(
                env,
                n_episodes=args.n_episodes,
                seed=args.seed,
                label="BASELINE",
                action_fn=lambda _obs: default_action,
            )

        rl_summary = _run_policy_eval(
            env,
            n_episodes=args.n_episodes,
            seed=args.seed + 10_000,
            label="RL",
            action_fn=lambda obs: model.predict(obs, deterministic=True)[0],
        )

        print("\n========================================")
        if baseline_summary is not None:
            throughput_gain = (
                (rl_summary["mean_throughput_msg_per_sec"] - baseline_summary["mean_throughput_msg_per_sec"])
                / max(baseline_summary["mean_throughput_msg_per_sec"], 1e-6)
                * 100.0
            )
            latency_p50_change = (
                (rl_summary["mean_latency_p50_ms"] - baseline_summary["mean_latency_p50_ms"])
                / max(baseline_summary["mean_latency_p50_ms"], 1e-6)
                * 100.0
            )
            latency_p95_change = (
                (rl_summary["mean_latency_p95_ms"] - baseline_summary["mean_latency_p95_ms"])
                / max(baseline_summary["mean_latency_p95_ms"], 1e-6)
                * 100.0
            )
            print("[对比结果] BASELINE vs RL")
            print(
                f"吞吐量: {baseline_summary['mean_throughput_msg_per_sec']:.2f} -> "
                f"{rl_summary['mean_throughput_msg_per_sec']:.2f} msg/s "
                f"({throughput_gain:+.2f}%)"
            )
            print(
                f"P50时延: {baseline_summary['mean_latency_p50_ms']:.2f} -> "
                f"{rl_summary['mean_latency_p50_ms']:.2f} ms ({latency_p50_change:+.2f}%)"
            )
            print(
                f"P95时延: {baseline_summary['mean_latency_p95_ms']:.2f} -> "
                f"{rl_summary['mean_latency_p95_ms']:.2f} ms ({latency_p95_change:+.2f}%)"
            )
            print(
                f"平均奖励: {baseline_summary['mean_reward']:.4f} -> "
                f"{rl_summary['mean_reward']:.4f}"
            )
        else:
            print("[RL结果]")
            print(json.dumps(rl_summary, indent=2, ensure_ascii=True))

        output_payload = {
            "baseline": baseline_summary,
            "rl": rl_summary,
            "n_episodes": args.n_episodes,
            "seed": args.seed,
        }
        if args.output_json:
            output_path = args.output_json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, indent=2, ensure_ascii=True)
            print(f"[评估] 结果已写入: {output_path}")
    finally:
        env.close()
        if workload is not None:
            workload.stop()


if __name__ == "__main__":
    main()
