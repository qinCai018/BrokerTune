import numpy as np
from gymnasium import spaces

from model.prioritized_nstep_replay_buffer import PrioritizedNStepReplayBuffer


def _add_transition(
    replay_buffer: PrioritizedNStepReplayBuffer,
    idx: int,
    reward: float = 1.0,
    done: bool = False,
    info: dict = None,
) -> None:
    obs = np.full((1, 10), fill_value=float(idx), dtype=np.float32)
    next_obs = np.full((1, 10), fill_value=float(idx + 1), dtype=np.float32)
    action = np.full((1, 2), fill_value=0.1 * idx, dtype=np.float32)
    replay_buffer.add(
        obs=obs,
        next_obs=next_obs,
        action=action,
        reward=np.array([reward], dtype=np.float32),
        done=np.array([1.0 if done else 0.0], dtype=np.float32),
        infos=[dict(info or {})],
    )


def _make_buffer(**kwargs) -> PrioritizedNStepReplayBuffer:
    return PrioritizedNStepReplayBuffer(
        buffer_size=64,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        device="cpu",
        **kwargs,
    )


def test_nstep_transition_and_gamma_accumulation():
    buffer = _make_buffer(use_per=True, use_n_step=True, n_step=3, gamma=0.9)

    for i in range(5):
        _add_transition(buffer, idx=i, reward=1.0, done=(i == 4), info={"step": i + 1})

    assert buffer.size() == 5
    assert buffer.nstep_transitions_added == 5

    expected_first_return = 1.0 + 0.9 + 0.9**2
    assert np.isclose(buffer.rewards[0, 0], expected_first_return, atol=1e-5)
    assert int(buffer.n_steps[0, 0]) == 3
    assert int(buffer.n_steps[3, 0]) == 2
    assert int(buffer.n_steps[4, 0]) == 1

    sample = buffer.sample(batch_size=3)
    assert sample.weights.shape == (3, 1)
    assert sample.n_steps.shape == (3, 1)
    assert sample.constraint_ratios.shape == (3, 1)
    assert len(sample.indices) == 3

    before_updates = buffer.per_priority_updates
    buffer.update_priorities(sample.indices, np.full((3,), 2.0, dtype=np.float32))
    assert buffer.per_priority_updates > before_updates


def test_done_semantics_respects_terminated_and_truncated_from_info():
    buffer = _make_buffer(use_per=False, use_n_step=True, n_step=3, gamma=0.9)

    _add_transition(buffer, idx=0, reward=1.0, done=False, info={"step": 1, "terminated": True})
    _add_transition(buffer, idx=1, reward=1.0, done=False, info={"step": 2, "truncated": True})

    assert buffer.size() == 2
    assert float(buffer.dones[0, 0]) == 1.0
    assert float(buffer.dones[1, 0]) == 1.0
    assert int(buffer.n_steps[0, 0]) == 1
    assert int(buffer.n_steps[1, 0]) == 1


def test_nstep_forced_flush_on_step_discontinuity():
    buffer = _make_buffer(use_per=False, use_n_step=True, n_step=3, gamma=0.9)

    _add_transition(buffer, idx=0, reward=1.0, info={"step": 1})
    _add_transition(buffer, idx=1, reward=2.0, info={"step": 2})
    _add_transition(buffer, idx=2, reward=4.0, info={"step": 10})

    assert buffer.nstep_forced_flushes == 1
    assert buffer.size() == 2
    assert int(buffer.n_steps[0, 0]) == 2
    assert float(buffer.dones[0, 0]) == 1.0
    assert np.isclose(float(buffer.rewards[0, 0]), 1.0 + 0.9 * 2.0, atol=1e-6)
    assert int(buffer.n_steps[1, 0]) == 1
    assert float(buffer.dones[1, 0]) == 1.0


def test_nstep_adaptive_halves_effective_horizon_when_done_ratio_high():
    buffer = _make_buffer(
        use_per=False,
        use_n_step=True,
        n_step=6,
        n_step_adaptive=True,
        n_step_adaptive_done_ratio_threshold=0.5,
        n_step_adaptive_window=4,
    )

    _add_transition(buffer, idx=0, done=True, info={"step": 1})
    _add_transition(buffer, idx=1, done=False, info={"step": 2})
    _add_transition(buffer, idx=2, done=True, info={"step": 3})
    _add_transition(buffer, idx=3, done=False, info={"step": 4})

    assert buffer.nstep_done_ratio >= 0.5
    assert int(buffer._current_effective_n) == 3


def test_per_beta_schedule_start_to_end():
    buffer = _make_buffer(
        use_per=True,
        use_n_step=False,
        per_beta0=0.2,
        per_beta_end=0.8,
        per_beta_anneal_steps=5,
    )

    for i in range(16):
        _add_transition(buffer, idx=i)

    _ = buffer.sample(batch_size=4)
    beta_after_first = buffer.get_current_beta()
    assert beta_after_first > 0.2

    for _ in range(20):
        _ = buffer.sample(batch_size=4)

    beta_final = buffer.get_current_beta()
    assert 0.79 <= beta_final <= 0.8


def test_per_priority_clip_floor_and_constraint_modulation():
    buffer = _make_buffer(
        use_per=True,
        use_n_step=False,
        per_alpha=1.0,
        per_eps=1e-3,
        per_clip_max=2.0,
        per_constraint_priority=True,
        per_constraint_scale=1.0,
    )

    _add_transition(buffer, idx=0)
    _add_transition(buffer, idx=1)

    # clip + floor
    buffer.update_priorities(
        np.array([0, 1], dtype=np.int64),
        np.array([100.0, 0.0], dtype=np.float32),
        constraint_signals=np.array([0.0, 0.0], dtype=np.float32),
    )
    assert buffer._sum_tree is not None
    assert buffer._sum_tree.get(0) <= 2.0 + 1e-8
    assert buffer._sum_tree.get(1) >= 1e-3

    # monotonic constraint modulation
    buffer.update_priorities(
        np.array([0, 1], dtype=np.int64),
        np.array([1.0, 1.0], dtype=np.float32),
        constraint_signals=np.array([0.0, 2.0], dtype=np.float32),
    )
    assert buffer._sum_tree.get(1) > buffer._sum_tree.get(0)


def test_mix_sampling_and_debug_stats_expose_weight_priority_distribution():
    np.random.seed(7)
    buffer = _make_buffer(
        use_per=True,
        use_n_step=False,
        per_mix_uniform_ratio=0.5,
        per_alpha=0.6,
        per_beta0=0.4,
    )

    for i in range(40):
        info = {
            "step": i + 1,
            "reward_components": {
                "latency_violation_ms": float(i % 3) * 10.0,
                "latency_limit_ms": 80.0,
                "unsafe": float((i % 3) > 0),
            },
        }
        _add_transition(buffer, idx=i, reward=1.0, info=info)

    td_errors = np.full((buffer.size(),), 0.01, dtype=np.float32)
    td_errors[3] = 10.0
    buffer.update_priorities(np.arange(buffer.size(), dtype=np.int64), td_errors)

    sampled = buffer.sample(batch_size=20)
    w = sampled.weights.detach().cpu().numpy().reshape(-1)

    assert sampled.constraint_ratios.shape == (20, 1)
    assert np.all(np.isfinite(w))
    assert np.all(w > 0.0)
    assert np.any(np.isclose(w, 1.0, atol=1e-6))
    assert np.any(w < 0.999)

    stats = buffer.get_debug_stats()
    required = {
        "per_last_priority_min",
        "per_last_priority_mean",
        "per_last_priority_p95",
        "per_last_priority_max",
        "per_last_weight_min",
        "per_last_weight_mean",
        "per_last_weight_max",
        "per_last_constraint_ratio",
    }
    assert required.issubset(set(stats.keys()))
