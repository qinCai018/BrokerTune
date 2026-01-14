#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的模型获取最优 Broker 配置并应用

这个脚本会：
1. 加载训练好的模型
2. 根据当前 Broker 状态，模型会输出最优的配置参数
3. 将这些配置应用到 Mosquitto Broker

使用示例：
    python3 script/apply_optimal_config.py \
        --model-path ./checkpoints/ddpg_mosquitto_final.zip \
        --apply-config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from environment import EnvConfig
from environment.knobs import apply_knobs
from tuner.utils import load_model, make_env


def get_optimal_config(model, env, apply: bool = False) -> dict:
    """
    使用训练好的模型获取最优配置
    
    Args:
        model: 训练好的 DDPG 模型
        env: 环境实例
        apply: 是否立即应用配置到 Broker
        
    Returns:
        最优配置字典
    """
    # 重置环境，获取当前状态
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result
    
    print("\n" + "=" * 80)
    print("当前 Broker 状态:")
    print("=" * 80)
    print(f"  连接数（归一化）: {obs[0]:.4f}")
    print(f"  消息速率（归一化）: {obs[1]:.4f}")
    print(f"  CPU 使用率: {obs[2]:.4f}")
    print(f"  内存使用率: {obs[3]:.4f}")
    print(f"  上下文切换率: {obs[4]:.4f}")
    print("=" * 80)
    
    # 使用模型预测最优动作（配置）
    action, _ = model.predict(obs, deterministic=True)
    
    # 将动作解码为配置参数
    knobs = env.knob_space.decode_action(action)
    
    print("\n" + "=" * 80)
    print("模型推荐的最优配置:")
    print("=" * 80)
    for key, value in knobs.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # 如果指定应用配置
    if apply:
        print("\n正在应用配置到 Broker...")
        try:
            used_restart = apply_knobs(knobs, dry_run=False)
            if used_restart:
                print("✅ 配置已应用（Broker 已重启）")
            else:
                print("✅ 配置已应用（Broker 已重载）")
        except Exception as e:
            print(f"❌ 应用配置失败: {e}")
            return knobs
    
    return knobs


def main():
    parser = argparse.ArgumentParser(
        description="使用训练好的模型获取并应用最优 Broker 配置"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="训练好的模型文件路径（.zip）",
    )
    parser.add_argument(
        "--apply-config",
        action="store_true",
        help="是否立即应用配置到 Broker（需要 sudo 权限）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="设备，例如 'cpu' 或 'cuda'（默认：cpu）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示配置，不实际应用（即使指定了 --apply-config）",
    )
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        print("\n可用的模型文件:")
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            for f in checkpoint_dir.glob("*.zip"):
                print(f"  - {f}")
        sys.exit(1)
    
    # 创建环境
    env_cfg = EnvConfig()
    env = make_env(env_cfg)
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    try:
        model = load_model(str(model_path), env, device=args.device)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 获取最优配置
    apply = args.apply_config and not args.dry_run
    knobs = get_optimal_config(model, env, apply=apply)
    
    # 显示配置摘要
    print("\n" + "=" * 80)
    print("配置摘要:")
    print("=" * 80)
    print(f"QoS 相关:")
    print(f"  - max_inflight_messages: {knobs.get('max_inflight_messages', 'N/A')}")
    print(f"  - max_inflight_bytes: {knobs.get('max_inflight_bytes', 'N/A')}")
    print(f"  - max_queued_messages: {knobs.get('max_queued_messages', 'N/A')}")
    print(f"  - max_queued_bytes: {knobs.get('max_queued_bytes', 'N/A')}")
    print(f"  - queue_qos0_messages: {knobs.get('queue_qos0_messages', 'N/A')}")
    print(f"\n内存相关:")
    print(f"  - memory_limit: {knobs.get('memory_limit', 'N/A')}")
    print(f"  - persistence: {knobs.get('persistence', 'N/A')}")
    print(f"  - autosave_interval: {knobs.get('autosave_interval', 'N/A')}")
    print(f"\n网络相关:")
    print(f"  - set_tcp_nodelay: {knobs.get('set_tcp_nodelay', 'N/A')}")
    print(f"\n协议相关:")
    print(f"  - max_packet_size: {knobs.get('max_packet_size', 'N/A')}")
    print(f"  - message_size_limit: {knobs.get('message_size_limit', 'N/A')}")
    print("=" * 80)
    
    if args.dry_run:
        print("\n[提示] 这是 dry-run 模式，配置未实际应用")
        print("       要实际应用配置，请使用: --apply-config（需要 sudo 权限）")
    elif not args.apply_config:
        print("\n[提示] 配置未应用。要应用配置，请使用: --apply-config（需要 sudo 权限）")
    
    env.close()


if __name__ == "__main__":
    main()
