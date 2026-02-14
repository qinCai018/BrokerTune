"""
训练/评估过程中通用的工具函数：
- 创建环境实例
- 创建 DDPG 模型（使用默认策略网络）
- 保存与加载模型
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
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
    replay_buffer_size: int = 1_000_000,
    learning_starts: int = 1_000,
    train_freq: int = 1,
    gradient_steps: int = 1,
    action_noise_type: str = "ou",
    action_noise_sigma: float = 0.2,
    action_noise_theta: float = 0.15,
    seed: Optional[int] = None,
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
        replay_buffer_size: Replay Buffer容量
        learning_starts: 多少步后开始梯度更新
        train_freq: 每多少步更新一次网络
        gradient_steps: 每次更新执行多少次梯度步
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

    model = DDPG(
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
        verbose=1,
        device=device,
        seed=seed,
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
