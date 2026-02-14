import numpy as np

from environment import EnvConfig
from environment.broker import MosquittoBrokerEnv


def _state(throughput_norm: float, latency_p50_norm: float) -> np.ndarray:
    arr = np.zeros(10, dtype=np.float32)
    arr[1] = throughput_norm
    arr[5] = latency_p50_norm
    arr[6] = latency_p50_norm * 1.5
    return arr


def test_reward_prefers_lower_latency_when_throughput_same():
    cfg = EnvConfig(
        reward_weight_base=0.6,
        reward_weight_step=0.2,
        reward_weight_latency_base=0.15,
        reward_weight_latency_step=0.05,
    )
    env = MosquittoBrokerEnv(cfg=cfg)

    initial = _state(throughput_norm=1.0, latency_p50_norm=0.50)
    prev = _state(throughput_norm=1.0, latency_p50_norm=0.50)
    better_latency = _state(throughput_norm=1.0, latency_p50_norm=0.20)
    worse_latency = _state(throughput_norm=1.0, latency_p50_norm=0.80)

    env._initial_state = initial
    env._throughput_history = [1.0, 1.0, 1.0]
    env._latency_history = [0.30, 0.25, 0.20]
    reward_low_latency = env._compute_reward(prev_state=prev, next_state=better_latency)

    env._latency_history = [0.70, 0.75, 0.80]
    reward_high_latency = env._compute_reward(prev_state=prev, next_state=worse_latency)

    assert reward_low_latency > reward_high_latency


def test_failure_transition_truncates_after_threshold():
    cfg = EnvConfig(max_consecutive_failures=2, failed_step_penalty=-2.5)
    env = MosquittoBrokerEnv(cfg=cfg)
    env._last_state = _state(throughput_norm=0.9, latency_p50_norm=0.3)

    _, reward1, _, truncated1, info1 = env._make_failure_transition(
        reason="sample_state_failed", error=RuntimeError("boom"), knobs={}
    )
    _, reward2, _, truncated2, info2 = env._make_failure_transition(
        reason="sample_state_failed", error=RuntimeError("boom"), knobs={}
    )

    assert reward1 == -2.5
    assert reward2 == -2.5
    assert truncated1 is False
    assert truncated2 is True
    assert info1["consecutive_failures"] == 1
    assert info2["consecutive_failures"] == 2
