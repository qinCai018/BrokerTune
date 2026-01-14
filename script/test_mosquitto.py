# -*- coding: utf-8 -*-
"""
对 Mosquitto Broker 进行一轮测试并记录数据。

使用示例（未训练的模型，用于观察训练过程）：
    from script.test_mosquitto import play
    from tuner.utils import make_env, make_ddpg_model
    from script.workload import WorkloadManager
    
    env = make_env()
    model = make_ddpg_model(env)  # 使用未训练的模型
    
    # 使用工作负载
    workload = WorkloadManager()
    workload.start(num_publishers=100, num_subscribers=100)
    
    # show=True 会显示详细的 state、action、reward
    state, action, reward, reward_sum = play(model, env, workload=workload, show=True)
    
    workload.stop()
    print(f"Total reward: {reward_sum}")

使用示例（已训练的模型）：
    from script.test_mosquitto import play
    from tuner.utils import make_env, load_model
    
    env = make_env()
    model = load_model("path/to/model.zip", env)  # 加载已训练的模型
    
    state, action, reward, reward_sum = play(model, env, show=True)
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径，以便导入模块
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

from environment import MosquittoBrokerEnv
from stable_baselines3 import DDPG

if TYPE_CHECKING:
    from script.workload import WorkloadManager


def play(
    model: DDPG,
    env: MosquittoBrokerEnv,
    show: bool = False,
    deterministic: bool = True,
    workload: Optional["WorkloadManager"] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    对 Mosquitto Broker 进行一轮测试并记录数据。
    
    Args:
        model: DDPG 模型（可以是未训练的模型）
        env: MosquittoBrokerEnv 环境实例
        show: 是否显示中间过程（显示每步的 state、action、reward）
        deterministic: 是否使用确定性策略（True）或带噪声的策略（False）
        workload: WorkloadManager 实例，如果提供则在测试期间运行工作负载
    
    Returns:
        state: 状态序列，shape (n_steps, state_dim)
        action: 动作序列，shape (n_steps, action_dim)
        reward: 奖励序列，shape (n_steps, 1)
        reward_sum: 总奖励
    """
    state = []
    action = []
    reward = []

    # 如果提供了工作负载管理器，确保工作负载正在运行
    if workload is not None and not workload.is_running():
        print("启动工作负载...")
        workload.start()

    # gymnasium兼容：reset返回(obs, info)元组
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        s, _ = reset_result
    else:
        s = reset_result
    
    done = False
    step_count = 0
    
    if show:
        print("\n" + "=" * 80)
        print("开始测试 - 观察强化学习训练过程")
        print("=" * 80)
        print(f"初始状态 (state_0): {s}")
        print(f"状态维度: {len(s)} (clients_norm, msg_rate_norm, cpu_ratio, mem_ratio, ctxt_ratio)")
        print("-" * 80)
    
    while not done:
        step_count += 1
        
        # 使用模型预测动作
        a, _ = model.predict(s, deterministic=deterministic)
        
        # 执行动作（兼容gymnasium的返回值格式）
        step_result = env.step(a)
        if len(step_result) == 4:
            ns, r, done, info = step_result
        else:
            # gymnasium v0.26+返回5个值：(obs, reward, terminated, truncated, info)
            ns, r, terminated, truncated, info = step_result
            done = terminated or truncated

        state.append(s)
        action.append(a)
        reward.append(r)

        if show:
            print(f"\n步骤 {step_count}:")
            print(f"  状态 (state_{step_count-1}):")
            print(f"    - clients_norm:     {s[0]:.6f}")
            print(f"    - msg_rate_norm:    {s[1]:.6f}")
            print(f"    - cpu_ratio:        {s[2]:.6f}")
            print(f"    - mem_ratio:        {s[3]:.6f}")
            print(f"    - ctxt_ratio:       {s[4]:.6f}")
            print(f"  动作 (action_{step_count-1}): {a}")
            print(f"  奖励 (reward_{step_count-1}): {r:.6f}")
            if "knobs" in info:
                print(f"  应用的配置 (knobs): {info['knobs']}")
            print(f"  是否结束: {done}")

        s = ns

    # 转换为 torch.Tensor
    state = torch.FloatTensor(np.array(state))
    action = torch.FloatTensor(np.array(action))
    reward = torch.FloatTensor(np.array(reward)).reshape(-1, 1)

    reward_sum = reward.sum().item()

    if show:
        print("\n" + "=" * 80)
        print("测试完成 - 汇总结果")
        print("=" * 80)
        print(f"总步数: {len(reward)}")
        print(f"总奖励: {reward_sum:.6f}")
        print(f"平均奖励: {reward_sum / len(reward) if len(reward) > 0 else 0:.6f}")
        print(f"最大奖励: {reward.max().item():.6f}")
        print(f"最小奖励: {reward.min().item():.6f}")
        print("\n状态序列形状:", state.shape)
        print("动作序列形状:", action.shape)
        print("奖励序列形状:", reward.shape)
        print("=" * 80)

    return state, action, reward, reward_sum


if __name__ == "__main__":
    # 示例使用
    from tuner.utils import make_env, make_ddpg_model, load_model
    from script.workload import WorkloadManager
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Mosquitto Broker with DDPG model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="模型文件路径（.zip）。如果不提供，将使用未训练的模型进行测试",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="设备，例如 'cpu' 或 'cuda'（默认：cpu）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="显示中间过程（默认：True，显示 state、action、reward）",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="使用确定性策略（默认：True）",
    )
    # 工作负载相关参数
    parser.add_argument(
        "--enable-workload",
        action="store_true",
        help="启用工作负载（使用 emqtt_bench）",
    )
    parser.add_argument(
        "--workload-publishers",
        type=int,
        default=100,
        help="发布者数量（默认：100）",
    )
    parser.add_argument(
        "--workload-subscribers",
        type=int,
        default=100,
        help="订阅者数量（默认：100）",
    )
    parser.add_argument(
        "--workload-topic",
        type=str,
        default="test/topic",
        help="MQTT 主题（默认：test/topic）",
    )
    parser.add_argument(
        "--workload-message-rate",
        type=int,
        default=100,
        help="每秒消息数（默认：100）",
    )
    parser.add_argument(
        "--workload-message-size",
        type=int,
        default=100,
        help="消息大小（字节，默认：100）",
    )
    parser.add_argument(
        "--emqtt-bench-path",
        type=str,
        default=None,
        help="emqtt_bench 可执行文件路径",
    )
    
    args = parser.parse_args()
    
    # 启用测试模式：不实际写入配置文件，只观察交互结果
    import os
    os.environ["BROKER_TUNER_DRY_RUN"] = "true"
    
    env = make_env()
    
    # 如果提供了模型路径，加载已训练的模型；否则创建未训练的模型用于测试
    if args.model_path:
        print(f"加载已训练的模型: {args.model_path}")
        model = load_model(args.model_path, env, device=args.device)
    else:
        print("使用未训练的模型进行测试（观察强化学习训练过程）")
        model = make_ddpg_model(env, device=args.device)
    
    # 创建工作负载管理器（如果启用）
    workload = None
    if args.enable_workload:
        print("\n" + "=" * 80)
        print("启动工作负载...")
        print("=" * 80)
        try:
            workload = WorkloadManager(
                broker_host=env.cfg.mqtt.host,
                broker_port=env.cfg.mqtt.port,
                emqtt_bench_path=args.emqtt_bench_path,
            )
            # 启动工作负载
            workload.start(
                num_publishers=args.workload_publishers,
                num_subscribers=args.workload_subscribers,
                topic=args.workload_topic,
                message_rate=args.workload_message_rate,
                message_size=args.workload_message_size,
            )
            print(f"[工作负载] 工作负载启动成功！")
            print(f"[工作负载] 可以使用以下命令监听消息:")
            print(f"[工作负载]   mosquitto_sub -h {env.cfg.mqtt.host} -t \"{args.workload_topic}\"")
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"\n[错误] 工作负载启动失败: {e}")
            print("\n提示:")
            print("1. 确保已安装 emqtt_bench:")
            print("   git clone https://github.com/emqx/emqtt-bench.git")
            print("   cd emqtt-bench && make")
            print("2. 或者设置 EMQTT_BENCH_PATH 环境变量指向 emqtt_bench 可执行文件")
            print("3. 或者使用 --emqtt-bench-path 参数指定路径\n")
            raise
    
    try:
        state, action, reward, reward_sum = play(
            model, env, 
            show=args.show, 
            deterministic=args.deterministic,
            workload=workload,
        )
        
        if not args.show:
            # 如果 show=False，在这里显示简要结果
            print(f"\n测试结果:")
            print(f"  总步数: {len(reward)}")
            print(f"  总奖励: {reward_sum:.6f}")
            print(f"  平均奖励: {reward_sum / len(reward) if len(reward) > 0 else 0:.6f}")
            print(f"  最大奖励: {reward.max().item():.6f}")
            print(f"  最小奖励: {reward.min().item():.6f}")
    finally:
        # 确保工作负载被停止
        if workload is not None:
            workload.stop()
        env.close()
