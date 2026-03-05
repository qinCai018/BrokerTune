"""
Feature-wise attention extractor for SB3 MlpPolicy.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeatureWiseAttentionExtractor(BaseFeaturesExtractor):
    """
    BrokerTune-oriented stable feature-wise gating for 1D vector observations:
      gate_raw = sigmoid(W2(relu(W1(x_norm))) / temperature)
      gate = clamp_min(gate_raw, gate_floor)
      gate = apply domain priors on critical dims (optional)
      effective_gate = residual_ratio + (1 - residual_ratio) * gate
      out = effective_gate * x
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        attention_hidden_dim: int = 64,
        attention_use_layer_norm: bool = False,
        attention_temperature: float = 1.2,
        attention_gate_floor: float = 0.03,
        attention_residual_ratio: float = 0.10,
        attention_use_domain_priors: bool = True,
        attention_critical_floor: float = 0.12,
        attention_critical_indices: Tuple[int, ...] = (1, 5, 6, 8, 9),
    ) -> None:
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("FeatureWiseAttentionExtractor only supports Box observation space.")
        if len(observation_space.shape) != 1:
            raise ValueError(
                f"FeatureWiseAttentionExtractor expects 1D observations, got shape={observation_space.shape}."
            )

        features_dim = int(observation_space.shape[0])
        super().__init__(observation_space=observation_space, features_dim=features_dim)

        hidden_dim = max(4, int(attention_hidden_dim))
        self.layer_norm: nn.Module = nn.LayerNorm(features_dim) if attention_use_layer_norm else nn.Identity()
        self.fc1 = nn.Linear(features_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, features_dim)

        self.attention_temperature = max(float(attention_temperature), 1e-6)
        self.attention_gate_floor = self._clip_01(attention_gate_floor)
        self.attention_residual_ratio = self._clip_01(attention_residual_ratio)
        self.attention_critical_floor = self._clip_01(attention_critical_floor)
        self._apply_domain_priors = bool(attention_use_domain_priors) and features_dim == 10
        self._active_critical_indices = tuple(
            int(idx) for idx in attention_critical_indices if 0 <= int(idx) < features_dim
        )

        self.last_gate: Optional[th.Tensor] = None
        self.last_gate_raw: Optional[th.Tensor] = None

    @staticmethod
    def _clip_01(value: float) -> float:
        return float(max(0.0, min(1.0, float(value))))

    def _sanitize(self, observations: th.Tensor) -> th.Tensor:
        return th.nan_to_num(observations, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_attention_weights(self, observations: th.Tensor) -> th.Tensor:
        x = self._sanitize(observations)
        x_norm = self.layer_norm(x)
        logits = self.fc2(th.relu(self.fc1(x_norm)))
        logits = logits / self.attention_temperature
        return th.sigmoid(logits)

    def _apply_brokertune_priors(self, gate: th.Tensor) -> th.Tensor:
        if not self._apply_domain_priors or not self._active_critical_indices:
            return gate
        gate = gate.clone()
        indices = list(self._active_critical_indices)
        gate[..., indices] = th.clamp_min(gate[..., indices], self.attention_critical_floor)
        return gate

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self._sanitize(observations)
        gate_raw = self.compute_attention_weights(x)
        gate = th.clamp_min(gate_raw, self.attention_gate_floor)
        gate = self._apply_brokertune_priors(gate)
        effective_gate = self.attention_residual_ratio + (1.0 - self.attention_residual_ratio) * gate

        self.last_gate_raw = gate_raw.detach()
        self.last_gate = effective_gate.detach()
        return x * effective_gate

    def get_gate_stats(self) -> Dict[str, float]:
        if self.last_gate is None:
            stats = {
                "gate_mean": 0.0,
                "gate_std": 0.0,
                "gate_min": 0.0,
                "gate_max": 0.0,
            }
            if self.last_gate_raw is not None:
                stats["gate_raw_mean"] = float(self.last_gate_raw.mean().item())
                stats["gate_raw_std"] = float(self.last_gate_raw.std().item())
                stats["gate_raw_min"] = float(self.last_gate_raw.min().item())
                stats["gate_raw_max"] = float(self.last_gate_raw.max().item())
            return stats

        gate = self.last_gate
        stats = {
            "gate_mean": float(gate.mean().item()),
            "gate_std": float(gate.std().item()),
            "gate_min": float(gate.min().item()),
            "gate_max": float(gate.max().item()),
        }
        if self.last_gate_raw is not None:
            gate_raw = self.last_gate_raw
            stats.update(
                {
                    "gate_raw_mean": float(gate_raw.mean().item()),
                    "gate_raw_std": float(gate_raw.std().item()),
                    "gate_raw_min": float(gate_raw.min().item()),
                    "gate_raw_max": float(gate_raw.max().item()),
                }
            )
        if self._active_critical_indices:
            critical = gate[..., list(self._active_critical_indices)]
            stats["gate_critical_mean"] = float(critical.mean().item())
        return stats
