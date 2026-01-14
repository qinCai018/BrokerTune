"""
模型评估入口：
- 加载已训练好的 DDPG 模型
- 在 MosquittoBrokerEnv 上运行若干 episode，统计平均奖励

使用示例：
    python -m tuner.evaluate --model-path ./checkpoints/ddpg_mosquitto_final.zip
"""

from __future__ import annotations

import argparse
from statistics import mean

import numpy as np

from environment import EnvConfig
from .utils import load_model, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained DDPG model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型文件路径（.zip）",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="评估 episode 数量",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="评估设备，例如 'cpu' 或 'cuda'（默认：cpu）",
    )
    return parser.parse_args()


def run_episode(model, env) -> float:
    # gymnasium兼容：reset返回(obs, info)元组
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result
    
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            # gymnasium v0.26+返回5个值：(obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        total_reward += float(reward)

    return total_reward


def main() -> None:
    args = parse_args()

    env_cfg = EnvConfig()
    env = make_env(env_cfg)

    model = load_model(args.model_path, env, device=args.device)

    rewards = []
    for i in range(args.n_episodes):
        ep_reward = run_episode(model, env)
        rewards.append(ep_reward)
        print(f"Episode {i + 1}/{args.n_episodes}: reward = {ep_reward:.3f}")

    print("========================================")
    print(f"Mean reward over {args.n_episodes} episodes: {mean(rewards):.3f}")
    print(f"Std reward: {np.std(rewards):.3f}")

    env.close()


if __name__ == "__main__":
    main()

