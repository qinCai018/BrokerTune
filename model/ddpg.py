"""
DDPG 算法的自定义网络结构和策略实现。

包含：
- CustomActor: 自定义 Actor 网络（策略网络）
- CustomCritic: 自定义 Critic 网络（Q值网络）
- CustomDDPGPolicy: 基于 stable-baselines3 DDPGPolicy 的自定义策略
"""

import torch
import torch.nn as nn
import numpy as np

# 尝试不同的导入方式以兼容不同版本的 stable_baselines3
try:
    from stable_baselines3.ddpg.policies import DDPGPolicy
except (ImportError, AttributeError) as e:
    # 检查实际可用的类
    try:
        import stable_baselines3.ddpg.policies as ddpg_policies
        # 查找可能的策略类（MlpPolicy 是常见的默认策略）
        if hasattr(ddpg_policies, 'MlpPolicy'):
            DDPGPolicy = ddpg_policies.MlpPolicy
        elif hasattr(ddpg_policies, 'CnnPolicy'):
            DDPGPolicy = ddpg_policies.CnnPolicy
        else:
            # 查找所有包含 Policy 的类
            available_classes = [name for name in dir(ddpg_policies) 
                               if 'Policy' in name and not name.startswith('_')]
            if available_classes:
                DDPGPolicy = getattr(ddpg_policies, available_classes[0])
            else:
                # 尝试使用 TD3Policy（DDPG 是 TD3 的特殊情况）
                try:
                    from stable_baselines3.td3.policies import TD3Policy as DDPGPolicy
                except ImportError:
                    raise ImportError(
                        f"无法导入 DDPGPolicy。错误: {e}\n"
                        f"stable_baselines3.ddpg.policies 可用类: {[x for x in dir(ddpg_policies) if not x.startswith('_')]}\n"
                        "尝试: pip install --upgrade stable-baselines3"
                    )
    except Exception as e2:
        raise ImportError(
            f"无法导入 DDPGPolicy。原始错误: {e}\n"
            f"检查错误: {e2}\n"
            "尝试: pip install --upgrade stable-baselines3"
        )






class CustomActor(nn.Module):
    """
    自定义 Actor 网络（策略网络），输出动作值。

    网络结构：
    - 输入：状态向量 (state_dim)
    - 输出：动作向量 (action_dim)，范围 [0, 1]（通过 Sigmoid）
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Linear(64, action_dim),
            nn.Sigmoid(),  # 输出范围 [0, 1]，对应动作空间的归一化
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: 状态张量，shape (batch_size, state_dim) 或 (state_dim,)
        Returns:
            动作张量，shape (batch_size, action_dim) 或 (action_dim,)
        """
        return self.net(obs)



class CustomCritic(nn.Module):
    """
    自定义 Critic 网络（Q值网络），评估状态-动作对的 Q 值。

    网络结构：
    - 状态和动作分别通过独立网络处理
    - 拼接后通过 Q 网络输出标量 Q 值
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
        )

        self.action_net = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.Tanh(),
        )

        self.q_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: 状态张量，shape (batch_size, state_dim)
            actions: 动作张量，shape (batch_size, action_dim)
        Returns:
            Q 值张量，shape (batch_size, 1)
        """
        s = self.state_net(obs)
        a = self.action_net(actions)
        x = torch.cat([s, a], dim=1)
        return self.q_net(x)


class CustomDDPGPolicy(DDPGPolicy):
    """
    自定义 DDPG 策略，使用 CustomActor 和 CustomCritic。
    这个类继承自 stable_baselines3.ddpg.policies.DDPGPolicy，
    只需要重写 make_actor 和 make_critic 方法来使用自定义的网络结构。

    使用方式：
        通过 tuner.utils.make_ddpg_model() 创建模型，而不是直接实例化此类。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_actor(self) -> CustomActor:
        """
        创建自定义 Actor 网络。

        Returns:
            CustomActor 实例，已移动到指定设备（CPU/GPU）
        """
        return CustomActor(
            self.observation_space.shape[0],
            self.action_space.shape[0],
        ).to(self.device)

    def make_critic(self) -> CustomCritic:
        """
        创建自定义 Critic 网络。

        Returns:
            CustomCritic 实例，已移动到指定设备（CPU/GPU）
        """
        return CustomCritic(
            self.observation_space.shape[0],
            self.action_space.shape[0],
        ).to(self.device)