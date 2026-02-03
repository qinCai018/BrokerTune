"""
训练/评估过程中通用的工具函数：
- 创建环境实例
- 创建 DDPG 模型（使用默认策略网络）
- 保存与加载模型
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np

from environment import EnvConfig, MosquittoBrokerEnv


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
) -> DDPG:
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
        
    Returns:
        DDPG模型实例
        
    注意：
        stable_baselines3 的 DDPG 不支持分别设置 actor 和 critic 的学习率。
        如果同时指定了 actor_lr 和 critic_lr，将使用两者的平均值作为统一的学习率。
    """
    n_actions = env.action_space.shape[0]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.2 * np.ones(n_actions),
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

    model = DDPG(
        policy="MlpPolicy",
        env=env,
        action_noise=action_noise,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        verbose=1,
        device=device,
    )
    return model


def save_model(model: DDPG, save_path: Union[str, Path]) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))


def load_model(load_path: Union[str, Path], env: MosquittoBrokerEnv, device: str = "cpu") -> DDPG:
    return DDPG.load(
        path=str(load_path),
        env=env,
        device=device,
    )
