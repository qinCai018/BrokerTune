"""
DDPG extension with PER/IS-weighted critic loss and N-step targets.
"""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DDPG
from stable_baselines3.common.utils import polyak_update


class EnhancedDDPG(DDPG):
    """
    Compatible with standard SB3 replay buffers and the custom
    PrioritizedNStepReplayBuffer.
    """

    def __init__(
        self,
        *args,
        utd_ratio: int = 1,
        policy_delay: int = 1,
        critic_loss: str = "mse",
        grad_clip_norm: float = 0.0,
        target_q_clip: float = 0.0,
        use_constraint_weighting: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.utd_ratio = max(1, int(utd_ratio))
        self.policy_delay = max(1, int(policy_delay))
        critic_loss_normalized = str(critic_loss).strip().lower()
        if critic_loss_normalized not in {"mse", "huber"}:
            raise ValueError(f"Unsupported critic_loss: {critic_loss}. Available: mse, huber")
        self.critic_loss = critic_loss_normalized
        self.grad_clip_norm = float(grad_clip_norm)
        self.target_q_clip = float(target_q_clip)
        self.use_constraint_weighting = bool(use_constraint_weighting)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        effective_gradient_steps = int(gradient_steps)
        if effective_gradient_steps > 0:
            effective_gradient_steps = int(effective_gradient_steps * self.utd_ratio)
        if effective_gradient_steps <= 0:
            return

        actor_losses = []
        critic_losses = []
        per_updates = 0
        beta_values = []
        gamma_n_values = []
        weight_means = []
        weight_mins = []
        weight_maxs = []
        constraint_ratio_values = []
        constraint_weight_means = []
        constraint_weighting_active_flags = []
        td_error_means = []
        td_error_p95_values = []
        td_error_max_values = []
        q_value_means = []
        q_value_max_values = []
        action_means = []
        action_mins = []
        action_maxs = []
        critic_grad_norms = []
        actor_grad_norms = []

        for _ in range(effective_gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            sample_weights = getattr(replay_data, "weights", None)
            if sample_weights is None:
                sample_weights = th.ones_like(replay_data.rewards)
            sample_weights = sample_weights.float()
            constraint_weights = th.ones_like(sample_weights)
            constraint_weighting_active = 0.0

            sampled_constraint_ratios = getattr(replay_data, "constraint_ratios", None)
            if self.use_constraint_weighting and sampled_constraint_ratios is not None:
                constraint_weights = 1.0 + sampled_constraint_ratios.float().clamp_min(0.0)
                constraint_weighting_active = 1.0

            loss_weights = sample_weights * constraint_weights

            sampled_n_steps = getattr(replay_data, "n_steps", None)
            if sampled_n_steps is None:
                discount = th.full_like(replay_data.rewards, self.gamma)
            else:
                discount = th.pow(th.full_like(replay_data.rewards, self.gamma), sampled_n_steps.float())
                gamma_n_values.append(float(discount.mean().item()))

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discount * next_q_values
                if self.target_q_clip > 0.0:
                    target_q_values = target_q_values.clamp(-self.target_q_clip, self.target_q_clip)

            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            if self.critic_loss == "huber":
                critic_loss = sum(
                    (F.smooth_l1_loss(current_q, target_q_values, reduction="none") * loss_weights).mean()
                    for current_q in current_q_values
                )
            else:
                critic_loss = sum((((current_q - target_q_values) ** 2) * loss_weights).mean() for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(float(critic_loss.item()))
            weight_means.append(float(sample_weights.mean().item()))
            weight_mins.append(float(sample_weights.min().item()))
            weight_maxs.append(float(sample_weights.max().item()))
            constraint_weight_means.append(float(constraint_weights.mean().item()))
            constraint_weighting_active_flags.append(constraint_weighting_active)

            td_error_tensor = (current_q_values[0] - target_q_values).detach().abs().reshape(-1)
            td_error_np = td_error_tensor.cpu().numpy()
            td_error_means.append(float(td_error_tensor.mean().item()))
            td_error_p95_values.append(float(np.percentile(td_error_np, 95)))
            td_error_max_values.append(float(td_error_tensor.max().item()))

            q_tensor = current_q_values[0].detach().reshape(-1)
            q_value_means.append(float(q_tensor.mean().item()))
            q_value_max_values.append(float(q_tensor.max().item()))

            action_tensor = replay_data.actions.detach().reshape(-1)
            action_means.append(float(action_tensor.mean().item()))
            action_mins.append(float(action_tensor.min().item()))
            action_maxs.append(float(action_tensor.max().item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_clip_norm > 0.0:
                critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
                critic_grad_norms.append(float(critic_grad_norm))
            self.critic.optimizer.step()

            if hasattr(replay_data, "indices") and hasattr(self.replay_buffer, "update_priorities"):
                constraint_ratios = getattr(replay_data, "constraint_ratios", None)
                constraint_signal = None
                if constraint_ratios is not None:
                    constraint_signal = constraint_ratios.detach().reshape(-1).cpu().numpy()
                    constraint_ratio_values.append(float((constraint_ratios > 0).float().mean().item()))

                if constraint_signal is not None:
                    try:
                        self.replay_buffer.update_priorities(  # type: ignore[attr-defined]
                            replay_data.indices,
                            td_error_np,
                            constraint_signals=constraint_signal,
                        )
                    except TypeError:
                        self.replay_buffer.update_priorities(replay_data.indices, td_error_np)  # type: ignore[attr-defined]
                else:
                    self.replay_buffer.update_priorities(replay_data.indices, td_error_np)  # type: ignore[attr-defined]
                per_updates += int(len(td_error_np))
                if hasattr(self.replay_buffer, "get_current_beta"):
                    beta_values.append(float(self.replay_buffer.get_current_beta()))  # type: ignore[attr-defined]

            if self._n_updates % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(float(actor_loss.item()))

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                if self.grad_clip_norm > 0.0:
                    actor_grad_norm = th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
                    actor_grad_norms.append(float(actor_grad_norm))
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/utd_ratio", float(self.utd_ratio), exclude="tensorboard")
        self.logger.record("train/effective_gradient_steps", float(effective_gradient_steps), exclude="tensorboard")
        self.logger.record("train/policy_delay", float(self.policy_delay), exclude="tensorboard")
        self.logger.record("train/critic_loss_type", self.critic_loss, exclude="tensorboard")
        self.logger.record("train/target_q_clip", float(self.target_q_clip), exclude="tensorboard")
        self.logger.record("train/use_constraint_weighting", float(self.use_constraint_weighting), exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        if critic_losses:
            self.logger.record("train/critic_loss", np.mean(critic_losses))

        if hasattr(self.replay_buffer, "n_step"):
            n_step = int(getattr(self.replay_buffer, "n_step"))
            use_n_step = bool(getattr(self.replay_buffer, "use_n_step", False))
            self.logger.record("train/n_step", n_step if use_n_step else 1)
            self.logger.record("train/gamma_n", float(self.gamma**n_step) if use_n_step else float(self.gamma))

        if gamma_n_values:
            self.logger.record("train/sampled_gamma_n_mean", float(np.mean(gamma_n_values)))
        if weight_means:
            self.logger.record("train/is_weight_mean", float(np.mean(weight_means)))
            self.logger.record("train/is_weight_min", float(np.mean(weight_mins)))
            self.logger.record("train/is_weight_max", float(np.mean(weight_maxs)))
        if per_updates > 0:
            self.logger.record("train/per_priority_updates", float(per_updates))
        if beta_values:
            self.logger.record("train/per_beta", float(np.mean(beta_values)))
        if constraint_ratio_values:
            self.logger.record("train/per_constraint_ratio", float(np.mean(constraint_ratio_values)))
        if constraint_weight_means:
            self.logger.record("train/constraint_weight_mean", float(np.mean(constraint_weight_means)))
        if constraint_weighting_active_flags:
            self.logger.record("train/constraint_weighting_active", float(np.mean(constraint_weighting_active_flags)))
        if td_error_means:
            self.logger.record("train/td_error_mean", float(np.mean(td_error_means)))
            self.logger.record("train/td_error_p95", float(np.mean(td_error_p95_values)))
            self.logger.record("train/td_error_max", float(np.mean(td_error_max_values)))
        if q_value_means:
            self.logger.record("train/q_value_mean", float(np.mean(q_value_means)))
            self.logger.record("train/q_value_max", float(np.mean(q_value_max_values)))
        if action_means:
            self.logger.record("train/action_mean", float(np.mean(action_means)))
            self.logger.record("train/action_min", float(np.mean(action_mins)))
            self.logger.record("train/action_max", float(np.mean(action_maxs)))
        if critic_grad_norms:
            self.logger.record("train/critic_grad_norm", float(np.mean(critic_grad_norms)))
        if actor_grad_norms:
            self.logger.record("train/actor_grad_norm", float(np.mean(actor_grad_norms)))
