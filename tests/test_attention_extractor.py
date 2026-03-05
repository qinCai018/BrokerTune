import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces

from model.attention_extractor import FeatureWiseAttentionExtractor
from tuner.utils import make_ddpg_model


def test_attention_extractor_keeps_shape_and_has_trainable_params():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    extractor = FeatureWiseAttentionExtractor(
        observation_space=obs_space,
        attention_hidden_dim=16,
        attention_use_layer_norm=True,
    )

    x = th.randn(4, 10)
    y = extractor(x)
    assert tuple(y.shape) == (4, 10)

    weights_raw = extractor.compute_attention_weights(x)
    assert tuple(weights_raw.shape) == (4, 10)
    assert extractor.last_gate_raw is not None
    assert th.all(weights_raw >= 0.0)
    assert th.all(weights_raw <= 1.0)

    param_count = sum(p.numel() for p in extractor.parameters())
    assert param_count > 0


def test_attention_residual_and_gate_floor_keep_minimum_gate():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    extractor = FeatureWiseAttentionExtractor(
        observation_space=obs_space,
        attention_hidden_dim=16,
        attention_use_layer_norm=False,
        attention_gate_floor=0.2,
        attention_residual_ratio=0.1,
        attention_use_domain_priors=False,
    )

    with th.no_grad():
        extractor.fc1.weight.zero_()
        extractor.fc1.bias.fill_(-100.0)
        extractor.fc2.weight.fill_(1.0)
        extractor.fc2.bias.fill_(-100.0)

    x = th.randn(4, 10)
    y = extractor(x)
    assert extractor.last_gate is not None
    expected_gate_min = 0.1 + (1.0 - 0.1) * 0.2
    assert th.all(extractor.last_gate >= expected_gate_min - 1e-6)
    assert th.allclose(y, x * extractor.last_gate, atol=1e-6, rtol=1e-6)


def test_attention_domain_priors_boost_critical_dims_for_brokertune_state():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    extractor = FeatureWiseAttentionExtractor(
        observation_space=obs_space,
        attention_hidden_dim=16,
        attention_use_layer_norm=False,
        attention_gate_floor=0.01,
        attention_residual_ratio=0.0,
        attention_use_domain_priors=True,
        attention_critical_floor=0.5,
        attention_critical_indices=(1, 5, 6, 8, 9),
    )

    with th.no_grad():
        extractor.fc1.weight.zero_()
        extractor.fc1.bias.fill_(-100.0)
        extractor.fc2.weight.fill_(1.0)
        extractor.fc2.bias.fill_(-100.0)

    _ = extractor(th.randn(3, 10))
    assert extractor.last_gate is not None
    gate = extractor.last_gate
    critical_indices = [1, 5, 6, 8, 9]
    normal_indices = [0, 2, 3, 4, 7]

    assert th.all(gate[:, critical_indices] >= 0.5 - 1e-6)
    assert th.all(gate[:, normal_indices] < 0.5)


def test_attention_gate_stats_are_complete_and_finite():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    extractor = FeatureWiseAttentionExtractor(
        observation_space=obs_space,
        attention_hidden_dim=16,
        attention_use_layer_norm=True,
        attention_use_domain_priors=True,
    )
    _ = extractor(th.randn(4, 10))
    stats = extractor.get_gate_stats()

    required = {
        "gate_mean",
        "gate_std",
        "gate_min",
        "gate_max",
        "gate_raw_mean",
        "gate_raw_std",
        "gate_raw_min",
        "gate_raw_max",
        "gate_critical_mean",
    }
    assert required.issubset(set(stats.keys()))
    for key in required:
        assert np.isfinite(stats[key])


def test_attention_checkpoint_load_is_strictly_compatible_with_legacy_keys():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    legacy_like = FeatureWiseAttentionExtractor(
        observation_space=obs_space,
        attention_hidden_dim=16,
        attention_use_layer_norm=True,
    )
    state_dict = legacy_like.state_dict()
    expected_keys = {
        "layer_norm.weight",
        "layer_norm.bias",
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
    }
    assert set(state_dict.keys()) == expected_keys

    upgraded = FeatureWiseAttentionExtractor(
        observation_space=obs_space,
        attention_hidden_dim=16,
        attention_use_layer_norm=True,
        attention_temperature=1.2,
        attention_gate_floor=0.03,
        attention_residual_ratio=0.1,
        attention_use_domain_priors=True,
    )
    load_info = upgraded.load_state_dict(state_dict, strict=True)
    assert load_info.missing_keys == []
    assert load_info.unexpected_keys == []


def test_make_ddpg_model_wires_attention_extractor_to_actor_and_critic():
    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

        def reset(self, **kwargs):
            return np.zeros(10, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(10, dtype=np.float32), 0.0, False, False, {}

    model = make_ddpg_model(
        env=DummyEnv(),
        use_attention=True,
        attention_hidden_dim=16,
        attention_use_layer_norm=True,
    )

    actor_extractor = getattr(model.policy.actor, "features_extractor", None)
    critic_extractor = getattr(model.policy.critic, "features_extractor", None)
    assert isinstance(actor_extractor, FeatureWiseAttentionExtractor)
    assert isinstance(critic_extractor, FeatureWiseAttentionExtractor)
