"""
训练/评估过程中通用的工具函数：
- 创建环境实例
- 创建 DDPG 模型（使用默认策略网络）
- 保存与加载模型
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
import numpy as np

from environment import EnvConfig, MosquittoBrokerEnv
from model import (
    EnhancedDDPG,
    FeatureWiseAttentionExtractor,
    PrioritizedNStepReplayBuffer,
)


def make_env(cfg: Optional[EnvConfig] = None, workload_manager: Optional[Any] = None) -> MosquittoBrokerEnv:
    """
    根据 EnvConfig 创建一个 MosquittoBrokerEnv 实例。
    
    Args:
        cfg: 环境配置
        workload_manager: 工作负载管理器（可选），如果提供，Broker重启后将自动重启工作负载
    """
    env_cfg = cfg or EnvConfig()
    env = MosquittoBrokerEnv(env_cfg, workload_manager=workload_manager)
    return env


def make_ddpg_model(
    env: MosquittoBrokerEnv,
    learning_rate: Union[float, dict] = 1e-4,
    batch_size: int = 128,
    gamma: float = 0.99,
    tau: float = 0.005,
    actor_lr: Optional[float] = None,
    critic_lr: Optional[float] = None,
    device: str = "cpu",
    replay_buffer_size: int = 1_000_000,
    learning_starts: int = 1_000,
    train_freq: int = 1,
    gradient_steps: int = 1,
    utd_ratio: int = 1,
    policy_delay: int = 1,
    critic_loss: str = "mse",
    grad_clip_norm: float = 0.0,
    target_q_clip: float = 0.0,
    use_constraint_weighting: bool = False,
    action_noise_type: str = "ou",
    action_noise_sigma: float = 0.2,
    action_noise_theta: float = 0.15,
    use_attention: bool = False,
    attention_hidden_dim: int = 64,
    attention_use_layer_norm: bool = False,
    use_per: bool = False,
    per_alpha: float = 0.6,
    per_beta0: float = 0.4,
    per_beta_end: float = 1.0,
    per_eps: float = 1e-6,
    per_clip_max: float = 0.0,
    per_mix_uniform_ratio: float = 0.0,
    per_constraint_priority: bool = False,
    per_constraint_scale: float = 1.0,
    per_beta_anneal_steps: int = 100_000,
    use_nstep: bool = False,
    n_step: int = 5,
    n_step_adaptive: bool = False,
    seed: Optional[int] = None,
) -> EnhancedDDPG:
    """
    使用默认 MlpPolicy 创建一个 DDPG 模型。
    
    Args:
        env: 环境实例
        learning_rate: 学习率（单个float值）。如果同时指定了actor_lr和critic_lr，此参数将被忽略
        batch_size: 批次大小
        gamma: 折扣因子
        tau: 目标网络软更新系数
        actor_lr: Actor学习率（可选）。如果同时指定了actor_lr和critic_lr，将使用两者的平均值
        critic_lr: Critic学习率（可选）。如果同时指定了actor_lr和critic_lr，将使用两者的平均值
        device: 训练设备
        replay_buffer_size: Replay Buffer容量
        learning_starts: 多少步后开始梯度更新
        train_freq: 每多少步更新一次网络
        gradient_steps: 每次更新执行多少次梯度步
        utd_ratio: UTD比率，每次训练调用额外执行倍率（effective_steps=gradient_steps*utd_ratio）
        policy_delay: actor/target延迟更新频率（每多少次critic更新执行1次actor更新）
        critic_loss: critic损失类型（mse/huber）
        grad_clip_norm: 梯度裁剪阈值（<=0表示关闭）
        target_q_clip: target Q裁剪阈值（<=0表示关闭）
        use_constraint_weighting: 是否启用约束感知loss加权
        action_noise_type: 探索噪声类型（ou/normal/none）
        action_noise_sigma: 动作噪声标准差
        action_noise_theta: OU噪声theta
        seed: 随机种子
        
    Returns:
        DDPG模型实例
        
    注意：
        stable_baselines3 的 DDPG 不支持分别设置 actor 和 critic 的学习率。
        如果同时指定了 actor_lr 和 critic_lr，将使用两者的平均值作为统一的学习率。
    """
    n_actions = env.action_space.shape[0]
    action_noise = None
    noise_type = action_noise_type.lower().strip()
    if noise_type == "ou":
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=float(action_noise_sigma) * np.ones(n_actions),
            theta=float(action_noise_theta),
        )
    elif noise_type in {"normal", "gaussian"}:
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=float(action_noise_sigma) * np.ones(n_actions),
        )
    elif noise_type in {"none", "off"}:
        action_noise = None
    else:
        raise ValueError(
            f"不支持的action_noise_type: {action_noise_type}，可选值: ou / normal / none"
        )

    # 如果指定了actor_lr或critic_lr，使用统一的学习率
    # 注意：stable_baselines3 的 DDPG 不支持分别设置 actor 和 critic 的学习率
    # 因此我们使用平均值或其中一个值作为统一的学习率
    if actor_lr is not None or critic_lr is not None:
        # 确定默认值
        default_lr = 1e-4
        if isinstance(learning_rate, dict):
            # 如果已经是字典，取平均值
            default_lr = (learning_rate.get("actor", 1e-4) + learning_rate.get("critic", 1e-4)) / 2
        elif isinstance(learning_rate, (int, float)):
            default_lr = float(learning_rate)
        
        # 计算最终学习率：如果两个都指定了，使用平均值；否则使用指定的那个
        if actor_lr is not None and critic_lr is not None:
            learning_rate = (actor_lr + critic_lr) / 2
        elif actor_lr is not None:
            learning_rate = actor_lr
        elif critic_lr is not None:
            learning_rate = critic_lr
        else:
            learning_rate = default_lr

    policy_kwargs = {}
    if use_attention:
        policy_kwargs["features_extractor_class"] = FeatureWiseAttentionExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "attention_hidden_dim": int(attention_hidden_dim),
            "attention_use_layer_norm": bool(attention_use_layer_norm),
        }

    replay_buffer_class = None
    replay_buffer_kwargs = None
    if use_per or use_nstep:
        replay_buffer_class = PrioritizedNStepReplayBuffer
        replay_buffer_kwargs = {
            "use_per": bool(use_per),
            "per_alpha": float(per_alpha),
            "per_beta0": float(per_beta0),
            "per_beta_end": float(per_beta_end),
            "per_eps": float(per_eps),
            "per_clip_max": float(per_clip_max),
            "per_mix_uniform_ratio": float(per_mix_uniform_ratio),
            "per_constraint_priority": bool(per_constraint_priority),
            "per_constraint_scale": float(per_constraint_scale),
            "per_beta_anneal_steps": int(per_beta_anneal_steps),
            "use_n_step": bool(use_nstep),
            "n_step": int(n_step),
            "n_step_adaptive": bool(n_step_adaptive),
            "gamma": float(gamma),
        }

    model = EnhancedDDPG(
        policy="MlpPolicy",
        env=env,
        action_noise=action_noise,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        buffer_size=int(replay_buffer_size),
        learning_starts=int(learning_starts),
        train_freq=(int(train_freq), "step"),
        gradient_steps=int(gradient_steps),
        utd_ratio=int(utd_ratio),
        policy_delay=int(policy_delay),
        critic_loss=str(critic_loss),
        grad_clip_norm=float(grad_clip_norm),
        target_q_clip=float(target_q_clip),
        use_constraint_weighting=bool(use_constraint_weighting),
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        verbose=1,
        device=device,
        seed=seed,
    )
    return model


def save_model(model: EnhancedDDPG, save_path: Union[str, Path]) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))


def load_model(load_path: Union[str, Path], env: MosquittoBrokerEnv, device: str = "cpu") -> EnhancedDDPG:
    try:
        return EnhancedDDPG.load(
            path=str(load_path),
            env=env,
            device=device,
        )
    except Exception:
        # 回退兼容历史模型
        from stable_baselines3 import DDPG

        return DDPG.load(
            path=str(load_path),
            env=env,
            device=device,
        )
