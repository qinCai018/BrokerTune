"""
BrokerTuner - 基于强化学习的 MQTT Broker 参数调优工具包

该包提供了使用 DDPG 算法训练和评估 MQTT Broker（Mosquitto）参数调优模型的工具。

主要模块：
- train: 模型训练入口
- evaluate: 模型评估入口
- utils: 工具函数（环境创建、模型创建、保存/加载等）
"""

from __future__ import annotations

# 导出常用的工具函数，方便直接使用
from .utils import (
    make_env,
    make_ddpg_model,
    save_model,
    load_model,
)

# 导出主要入口函数
from .train import main as train_main
from .evaluate import main as evaluate_main

__all__ = [
    # 工具函数
    "make_env",
    "make_ddpg_model",
    "save_model",
    "load_model",
    # 入口函数
    "train_main",
    "evaluate_main",
]
