import numpy as np

from tuner.evaluate import _extract_metrics, summarize


def test_extract_metrics_prefers_info_values():
    obs = np.zeros(10, dtype=np.float32)
    obs[1] = 1.0
    obs[5] = 0.5
    obs[6] = 0.8
    info = {
        "throughput_norm": 2.0,
        "throughput_msg_per_sec": 12345.0,
        "latency_p50_ms": 22.0,
        "latency_p95_ms": 88.0,
    }
    metrics = _extract_metrics(obs, info)
    assert metrics["throughput_norm"] == 2.0
    assert metrics["throughput_msg_per_sec"] == 12345.0
    assert metrics["latency_p50_ms"] == 22.0
    assert metrics["latency_p95_ms"] == 88.0


def test_summarize_returns_mean_and_std():
    summary = summarize(
        [
            {
                "episode_reward": 1.0,
                "episode_throughput_msg_per_sec": 100.0,
                "episode_latency_p50_ms": 10.0,
                "episode_latency_p95_ms": 20.0,
            },
            {
                "episode_reward": 3.0,
                "episode_throughput_msg_per_sec": 300.0,
                "episode_latency_p50_ms": 30.0,
                "episode_latency_p95_ms": 40.0,
            },
        ]
    )
    assert summary["mean_reward"] == 2.0
    assert summary["mean_throughput_msg_per_sec"] == 200.0
    assert summary["mean_latency_p50_ms"] == 20.0
    assert summary["mean_latency_p95_ms"] == 30.0
    assert summary["std_reward"] > 0.0
