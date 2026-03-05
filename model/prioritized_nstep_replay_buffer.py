"""
Replay buffer with optional N-step aggregation and Prioritized Experience Replay.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

try:
    from stable_baselines3.common.vec_env import VecNormalize
except Exception:  # pragma: no cover - typing fallback
    VecNormalize = Any  # type: ignore


class PrioritizedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    indices: np.ndarray
    weights: th.Tensor
    n_steps: th.Tensor
    constraint_ratios: th.Tensor


class _BinaryTree:
    """
    Compact binary tree used for sum/min segment operations.
    """

    def __init__(self, capacity: int, op: str):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if op not in {"sum", "min"}:
            raise ValueError("op must be 'sum' or 'min'")
        self.capacity = int(capacity)
        self.op = op
        if op == "sum":
            self.tree = np.zeros(2 * self.capacity, dtype=np.float64)
            self.default = 0.0
        else:
            self.tree = np.full(2 * self.capacity, np.inf, dtype=np.float64)
            self.default = np.inf

    def _merge(self, left: float, right: float) -> float:
        if self.op == "sum":
            return float(left + right)
        return float(min(left, right))

    def update(self, index: int, value: float) -> None:
        idx = int(index) + self.capacity
        self.tree[idx] = value
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self._merge(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def get(self, index: int) -> float:
        return float(self.tree[int(index) + self.capacity])

    def query(self) -> float:
        return float(self.tree[1])

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        if self.op != "sum":
            raise RuntimeError("find_prefixsum_idx only supports sum tree")
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] >= prefixsum:
                idx = left
            else:
                prefixsum -= self.tree[left]
                idx = left + 1
        return idx - self.capacity


@dataclass
class _NStepTransition:
    obs: np.ndarray
    next_obs: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class PrioritizedNStepReplayBuffer(ReplayBuffer):
    """
    SB3-compatible replay buffer with:
      - optional N-step transition aggregation
      - optional PER (sum-tree sampling + IS weights)
    """

    def __init__(
        self,
        *args: Any,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta0: float = 0.4,
        per_beta_end: float = 1.0,
        per_eps: float = 1e-6,
        per_beta_anneal_steps: int = 100_000,
        per_clip_max: float = 0.0,
        per_mix_uniform_ratio: float = 0.0,
        per_constraint_priority: bool = False,
        per_constraint_scale: float = 1.0,
        use_n_step: bool = False,
        n_step: int = 5,
        n_step_adaptive: bool = False,
        n_step_adaptive_done_ratio_threshold: float = 0.35,
        n_step_adaptive_window: int = 128,
        gamma: float = 0.99,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1:
            raise NotImplementedError("PrioritizedNStepReplayBuffer currently supports n_envs=1 only")

        self.use_per = bool(use_per)
        self.per_alpha = float(per_alpha)
        self.per_beta0 = float(per_beta0)
        self.per_beta_start = float(per_beta0)
        self.per_beta_end = float(per_beta_end)
        self.per_eps = max(float(per_eps), 1e-12)
        self.per_beta_anneal_steps = max(1, int(per_beta_anneal_steps))
        self.per_clip_max = float(per_clip_max)
        self.per_mix_uniform_ratio = float(np.clip(float(per_mix_uniform_ratio), 0.0, 1.0))
        self.per_constraint_priority = bool(per_constraint_priority)
        self.per_constraint_scale = float(per_constraint_scale)

        self.use_n_step = bool(use_n_step)
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.n_step_adaptive = bool(n_step_adaptive)
        self.n_step_adaptive_done_ratio_threshold = float(
            np.clip(float(n_step_adaptive_done_ratio_threshold), 0.0, 1.0)
        )
        self.n_step_adaptive_window = max(1, int(n_step_adaptive_window))

        self.n_steps = np.ones((self.buffer_size, self.n_envs), dtype=np.int32)
        self._nstep_queue: Deque[_NStepTransition] = deque(maxlen=self.n_step)
        self._recent_done_flags: Deque[float] = deque(maxlen=self.n_step_adaptive_window)
        self._current_effective_n = self.n_step
        self._last_transition_step: Optional[int] = None

        self._sum_tree = _BinaryTree(self.buffer_size, op="sum") if self.use_per else None
        self._min_tree = _BinaryTree(self.buffer_size, op="min") if self.use_per else None
        self._max_priority = 1.0

        self._beta_step = 0
        self._current_beta = self.per_beta_start

        self.nstep_transitions_added = 0
        self.nstep_forced_flushes = 0
        self.nstep_done_ratio = 0.0
        self.nstep_last_forced_flush_reason = ""

        self.per_priority_updates = 0
        self.per_samples = 0

        self.last_sampled_weight_min = 1.0
        self.last_sampled_weight_mean = 1.0
        self.last_sampled_weight_max = 1.0

        self.last_sampled_priority_min = 1.0
        self.last_sampled_priority_mean = 1.0
        self.last_sampled_priority_p95 = 1.0
        self.last_sampled_priority_max = 1.0

        self.last_sampled_constraint_ratio = 0.0

    @staticmethod
    def _safe_bool(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer, float, np.floating)):
            return bool(float(value) > 0.0)
        return False

    @staticmethod
    def _extract_transition_step(info: Dict[str, Any]) -> Optional[int]:
        value = info.get("step")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _update_effective_n(self) -> None:
        if not self.use_n_step:
            self._current_effective_n = 1
            self.nstep_done_ratio = 0.0
            return
        if not self.n_step_adaptive:
            self._current_effective_n = self.n_step
            if self._recent_done_flags:
                self.nstep_done_ratio = float(np.mean(np.asarray(self._recent_done_flags, dtype=np.float32)))
            else:
                self.nstep_done_ratio = 0.0
            return

        if self._recent_done_flags:
            done_ratio = float(np.mean(np.asarray(self._recent_done_flags, dtype=np.float32)))
        else:
            done_ratio = 0.0
        self.nstep_done_ratio = done_ratio

        if done_ratio >= self.n_step_adaptive_done_ratio_threshold:
            self._current_effective_n = max(1, self.n_step // 2)
        else:
            self._current_effective_n = self.n_step

    def _extract_single_transition(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> _NStepTransition:
        obs_arr = np.array(obs[0], copy=True)
        next_obs_arr = np.array(next_obs[0], copy=True)
        action_arr = np.array(action[0], copy=True) if np.asarray(action).ndim > 1 else np.array(action, copy=True)
        reward_value = float(np.asarray(reward).reshape(-1)[0])
        done_from_array = bool(np.asarray(done).reshape(-1)[0] > 0.5)
        info_value = dict(infos[0]) if infos and isinstance(infos[0], dict) else {}
        done_from_info = self._safe_bool(info_value.get("done", False))
        terminated = self._safe_bool(info_value.get("terminated", False))
        truncated = self._safe_bool(info_value.get("truncated", False))
        done_value = bool(done_from_array or done_from_info or terminated or truncated)

        return _NStepTransition(
            obs=obs_arr,
            next_obs=next_obs_arr,
            action=action_arr,
            reward=reward_value,
            done=done_value,
            info=info_value,
        )

    def _store_single_transition(
        self,
        transition: _NStepTransition,
        n_steps: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        obs_arr = np.expand_dims(np.asarray(transition.obs), axis=0)
        next_obs_arr = np.expand_dims(np.asarray(next_obs), axis=0)
        action_arr = np.expand_dims(np.asarray(transition.action), axis=0)
        reward_arr = np.array([reward], dtype=np.float32)
        done_arr = np.array([float(done)], dtype=np.float32)
        info_arr = [dict(transition.info)]

        super().add(obs_arr, next_obs_arr, action_arr, reward_arr, done_arr, info_arr)

        index = (self.pos - 1) % self.buffer_size
        self.n_steps[index, 0] = int(n_steps)

        if self.use_per:
            assert self._sum_tree is not None
            assert self._min_tree is not None
            self._sum_tree.update(index, self._max_priority)
            self._min_tree.update(index, self._max_priority)

        if self.use_n_step:
            self.nstep_transitions_added += 1

    def _build_n_step_from_queue(
        self,
        max_steps: Optional[int] = None,
        force_done: bool = False,
    ) -> Tuple[_NStepTransition, int, float, np.ndarray, bool]:
        first = self._nstep_queue[0]
        cumulative_reward = 0.0
        steps = 0
        final_next_obs = first.next_obs
        final_done = False
        max_steps = self.n_step if max_steps is None else max(1, int(max_steps))

        for idx, item in enumerate(self._nstep_queue):
            cumulative_reward += (self.gamma ** idx) * float(item.reward)
            final_next_obs = item.next_obs
            steps = idx + 1
            if item.done:
                final_done = True
                break
            if steps >= max_steps:
                break

        if force_done and not final_done:
            final_done = True

        return first, steps, cumulative_reward, final_next_obs, final_done

    def _flush_one_n_step(self, max_steps: Optional[int] = None, force_done: bool = False) -> None:
        if not self._nstep_queue:
            return
        first, steps, cumulative_reward, final_next_obs, final_done = self._build_n_step_from_queue(
            max_steps=max_steps,
            force_done=force_done,
        )
        self._store_single_transition(
            transition=first,
            n_steps=steps,
            reward=cumulative_reward,
            next_obs=final_next_obs,
            done=final_done,
        )
        self._nstep_queue.popleft()

    def _flush_all_n_step(self, max_steps: Optional[int] = None, force_done: bool = False) -> None:
        while self._nstep_queue:
            self._flush_one_n_step(max_steps=max_steps, force_done=force_done)

    def _extract_constraint_ratio(self, info: Any) -> float:
        if not isinstance(info, dict):
            return 0.0

        reward_components = info.get("reward_components")
        if isinstance(reward_components, dict):
            latency_violation_ms = reward_components.get("latency_violation_ms", info.get("latency_violation_ms", 0.0))
            latency_limit_ms = reward_components.get("latency_limit_ms", info.get("latency_limit_ms", 0.0))
            unsafe = reward_components.get("unsafe", info.get("unsafe", False))
        else:
            latency_violation_ms = info.get("latency_violation_ms", 0.0)
            latency_limit_ms = info.get("latency_limit_ms", 0.0)
            unsafe = info.get("unsafe", False)

        try:
            violation = float(latency_violation_ms)
        except (TypeError, ValueError):
            violation = 0.0

        try:
            limit = float(latency_limit_ms)
        except (TypeError, ValueError):
            limit = 0.0

        if np.isfinite(violation) and np.isfinite(limit) and limit > self.per_eps:
            ratio = max(0.0, violation / limit)
            return float(ratio)

        if self._safe_bool(unsafe):
            return 1.0
        return 0.0

    def _extract_constraint_ratios(self, batch_inds: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        ratios = np.zeros(len(batch_inds), dtype=np.float32)
        info_store = getattr(self, "infos", None)
        if info_store is None:
            return ratios

        for i, (idx, env_idx) in enumerate(zip(batch_inds, env_indices)):
            try:
                info_value = info_store[int(idx), int(env_idx)]
            except Exception:
                info_value = None
            ratios[i] = float(self._extract_constraint_ratio(info_value))
        return ratios

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        transition = self._extract_single_transition(obs, next_obs, action, reward, done, infos)

        if not self.use_n_step:
            self._store_single_transition(
                transition=transition,
                n_steps=1,
                reward=transition.reward,
                next_obs=transition.next_obs,
                done=transition.done,
            )
            return

        transition_step = self._extract_transition_step(transition.info)
        if (
            self._nstep_queue
            and transition_step is not None
            and self._last_transition_step is not None
            and (transition_step <= self._last_transition_step or transition_step > (self._last_transition_step + 1))
        ):
            self._flush_all_n_step(max_steps=self._current_effective_n, force_done=True)
            self.nstep_forced_flushes += 1
            self.nstep_last_forced_flush_reason = f"step_discontinuity:{self._last_transition_step}->{transition_step}"

        self._nstep_queue.append(transition)
        self._recent_done_flags.append(1.0 if transition.done else 0.0)
        self._update_effective_n()

        if len(self._nstep_queue) >= self._current_effective_n:
            self._flush_one_n_step(max_steps=self._current_effective_n, force_done=False)

        if transition.done:
            self._flush_all_n_step(max_steps=self._current_effective_n, force_done=False)

        self._last_transition_step = transition_step

    def _sample_indices_uniform(self, batch_size: int) -> np.ndarray:
        if not self.optimize_memory_usage:
            upper = self.buffer_size if self.full else self.pos
            return np.random.randint(0, upper, size=batch_size)
        if self.full:
            return (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        return np.random.randint(0, self.pos, size=batch_size)

    def _sample_indices_per(self, batch_size: int) -> np.ndarray:
        assert self._sum_tree is not None
        total_priority = self._sum_tree.query()
        if total_priority <= 0:
            return self._sample_indices_uniform(batch_size)

        segment = total_priority / float(batch_size)
        batch_inds = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            mass = np.random.uniform(low, high)
            batch_inds[i] = self._sum_tree.find_prefixsum_idx(mass)
        return batch_inds

    def _get_samples_with_meta(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> Tuple[ReplayBufferSamples, np.ndarray]:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )

        samples = ReplayBufferSamples(*tuple(map(self.to_torch, data)))
        return samples, env_indices

    def _compute_beta(self) -> float:
        progress = min(1.0, self._beta_step / float(self.per_beta_anneal_steps))
        beta = self.per_beta_start + progress * (self.per_beta_end - self.per_beta_start)
        lower = min(self.per_beta_start, self.per_beta_end)
        upper = max(self.per_beta_start, self.per_beta_end)
        return float(np.clip(beta, lower, upper))

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> PrioritizedReplayBufferSamples:
        if self.size() == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        self._beta_step += 1
        self._current_beta = self._compute_beta()

        if self.use_per:
            assert self._sum_tree is not None
            assert self._min_tree is not None

            uniform_ratio = float(np.clip(self.per_mix_uniform_ratio, 0.0, 1.0))
            n_uniform = int(round(batch_size * uniform_ratio))
            n_uniform = min(max(n_uniform, 0), batch_size)
            n_per = int(batch_size - n_uniform)

            per_inds = self._sample_indices_per(n_per) if n_per > 0 else np.empty((0,), dtype=np.int64)
            uniform_inds = self._sample_indices_uniform(n_uniform) if n_uniform > 0 else np.empty((0,), dtype=np.int64)

            batch_inds = np.concatenate([per_inds, uniform_inds], axis=0).astype(np.int64)
            source_is_per = np.concatenate(
                [
                    np.ones(len(per_inds), dtype=np.int8),
                    np.zeros(len(uniform_inds), dtype=np.int8),
                ],
                axis=0,
            )

            if len(batch_inds) != batch_size:
                if len(batch_inds) == 0:
                    batch_inds = self._sample_indices_uniform(batch_size)
                    source_is_per = np.zeros(batch_size, dtype=np.int8)
                else:
                    fill = self._sample_indices_uniform(batch_size - len(batch_inds))
                    batch_inds = np.concatenate([batch_inds, fill], axis=0)
                    source_is_per = np.concatenate([source_is_per, np.zeros(len(fill), dtype=np.int8)], axis=0)

            if batch_size > 1:
                perm = np.random.permutation(batch_size)
                batch_inds = batch_inds[perm]
                source_is_per = source_is_per[perm]

            replay_data, env_indices = self._get_samples_with_meta(batch_inds, env=env)
            priorities = np.array([self._sum_tree.get(int(idx)) for idx in batch_inds], dtype=np.float64)
            weights = np.ones(batch_size, dtype=np.float64)

            per_mask = source_is_per.astype(bool)
            if np.any(per_mask):
                total_priority = max(self._sum_tree.query(), self.per_eps)
                probs = np.clip(priorities[per_mask] / total_priority, self.per_eps, None)
                buffer_size = max(1, self.size())
                per_weights = np.power(buffer_size * probs, -self._current_beta)

                min_priority = self._min_tree.query()
                if not np.isfinite(min_priority) or min_priority <= 0:
                    max_weight = float(np.max(per_weights))
                else:
                    min_prob = max(min_priority / total_priority, self.per_eps)
                    max_weight = float(np.power(buffer_size * min_prob, -self._current_beta))
                per_weights = per_weights / max(max_weight, self.per_eps)
                weights[per_mask] = per_weights

            weights = np.clip(weights, self.per_eps, np.inf)

            self.per_samples += int(batch_size)
            self.last_sampled_weight_min = float(np.min(weights))
            self.last_sampled_weight_mean = float(np.mean(weights))
            self.last_sampled_weight_max = float(np.max(weights))

            self.last_sampled_priority_min = float(np.min(priorities))
            self.last_sampled_priority_mean = float(np.mean(priorities))
            self.last_sampled_priority_p95 = float(np.percentile(priorities, 95))
            self.last_sampled_priority_max = float(np.max(priorities))
        else:
            batch_inds = self._sample_indices_uniform(batch_size)
            replay_data, env_indices = self._get_samples_with_meta(batch_inds, env=env)
            weights = np.ones(batch_size, dtype=np.float64)

            self.last_sampled_weight_min = 1.0
            self.last_sampled_weight_mean = 1.0
            self.last_sampled_weight_max = 1.0

            self.last_sampled_priority_min = 1.0
            self.last_sampled_priority_mean = 1.0
            self.last_sampled_priority_p95 = 1.0
            self.last_sampled_priority_max = 1.0

        sampled_n_steps = self.n_steps[batch_inds, env_indices].reshape(-1, 1)
        sampled_constraint_ratios = self._extract_constraint_ratios(batch_inds, env_indices).reshape(-1, 1)
        self.last_sampled_constraint_ratio = float(np.mean(sampled_constraint_ratios > 0.0))

        return PrioritizedReplayBufferSamples(
            observations=replay_data.observations,
            actions=replay_data.actions,
            next_observations=replay_data.next_observations,
            dones=replay_data.dones,
            rewards=replay_data.rewards,
            indices=batch_inds,
            weights=self.to_torch(weights.reshape(-1, 1).astype(np.float32)),
            n_steps=self.to_torch(sampled_n_steps.reshape(-1, 1)),
            constraint_ratios=self.to_torch(sampled_constraint_ratios.astype(np.float32)),
        )

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
        constraint_signals: Optional[np.ndarray] = None,
    ) -> None:
        if not self.use_per:
            return
        assert self._sum_tree is not None
        assert self._min_tree is not None

        idx_arr = np.asarray(indices).reshape(-1).astype(np.int64)
        err_arr = np.asarray(td_errors).reshape(-1)
        constraint_arr = None
        if constraint_signals is not None:
            constraint_arr = np.asarray(constraint_signals).reshape(-1)

        n = min(len(idx_arr), len(err_arr))
        if n <= 0:
            return

        for i, (idx, err) in enumerate(zip(idx_arr[:n], err_arr[:n])):
            base = abs(float(err)) + self.per_eps
            if self.per_constraint_priority and constraint_arr is not None and i < len(constraint_arr):
                try:
                    constraint_ratio = max(0.0, float(constraint_arr[i]))
                except (TypeError, ValueError):
                    constraint_ratio = 0.0
                weight_constraint = 1.0 + self.per_constraint_scale * constraint_ratio
            else:
                weight_constraint = 1.0

            adjusted_priority = base * weight_constraint
            if self.per_clip_max > 0.0:
                adjusted_priority = min(adjusted_priority, self.per_clip_max)
            adjusted_priority = max(adjusted_priority, self.per_eps)

            priority = float(adjusted_priority ** self.per_alpha)
            priority = max(priority, self.per_eps ** self.per_alpha)
            self._sum_tree.update(int(idx), priority)
            self._min_tree.update(int(idx), priority)
            self._max_priority = max(self._max_priority, priority)

        self.per_priority_updates += int(n)

    def get_current_beta(self) -> float:
        return float(self._current_beta)

    def get_debug_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {
            "nstep_transitions_added": float(self.nstep_transitions_added),
            "n_step": float(self.n_step if self.use_n_step else 1),
            "nstep_effective_n": float(self._current_effective_n if self.use_n_step else 1),
            "nstep_done_ratio": float(self.nstep_done_ratio),
            "nstep_forced_flushes": float(self.nstep_forced_flushes),
            "per_beta": float(self._current_beta),
            "per_priority_updates": float(self.per_priority_updates),
            "per_samples": float(self.per_samples),
            "per_last_weight_min": float(self.last_sampled_weight_min),
            "per_last_weight_mean": float(self.last_sampled_weight_mean),
            "per_last_weight_max": float(self.last_sampled_weight_max),
            "per_last_priority_min": float(self.last_sampled_priority_min),
            "per_last_priority_mean": float(self.last_sampled_priority_mean),
            "per_last_priority_p95": float(self.last_sampled_priority_p95),
            "per_last_priority_max": float(self.last_sampled_priority_max),
            "per_last_constraint_ratio": float(self.last_sampled_constraint_ratio),
        }
        if self.use_per and self._sum_tree is not None and self._min_tree is not None:
            stats["per_total_priority"] = float(self._sum_tree.query())
            min_priority = self._min_tree.query()
            stats["per_min_priority"] = float(min_priority if np.isfinite(min_priority) else 0.0)
            stats["per_max_priority"] = float(self._max_priority)
        return stats
