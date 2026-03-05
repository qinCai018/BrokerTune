"""
模型训练入口：
- 创建 MosquittoBrokerEnv 环境
- 使用默认 MlpPolicy 的 DDPG 进行训练
- 定期保存模型

使用示例：
    python -m tuner.train --total-timesteps 100000 --save-dir ./checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import signal
import sys
import time
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
import numpy as np

# 导入gym/gymnasium用于包装类继承
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("警告: tqdm 未安装，将无法显示进度条。安装命令: pip install tqdm")

from environment import EnvConfig
from .utils import make_ddpg_model, make_env, save_model

# 尝试导入工作负载管理器
try:
    import sys
    from pathlib import Path
    # 添加 script 目录到路径
    script_dir = Path(__file__).parent.parent / "script"
    if script_dir.exists():
        sys.path.insert(0, str(script_dir.parent))
    from script.workload import WorkloadManager
    WORKLOAD_AVAILABLE = True
except ImportError:
    WORKLOAD_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPG for Mosquitto Broker tuning")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5_000_000,
        help="总训练步数（与 env.step 次数相同），默认：5,000,000",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="模型保存目录",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10_000,
        help="每隔多少步保存一次 checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="训练设备，例如 'cpu' 或 'cuda'（默认：cpu）",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="目标网络软更新系数，默认：0.005",
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=0.0001,
        help="Actor学习率，默认：0.0001",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=0.001,
        help="Critic学习率，默认：0.001",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="折扣因子，默认：0.99",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="训练批次大小，默认：128",
    )
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=200000,
        help="Replay Buffer容量，默认：200000",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1000,
        help="多少步后开始梯度更新，默认：1000",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="每多少环境步执行一次训练更新，默认：1",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="每次更新执行的梯度步数，默认：1",
    )
    parser.add_argument(
        "--utd_ratio",
        "--utd-ratio",
        type=int,
        default=1,
        help="UTD比率：每次训练调用额外执行倍率（effective_steps=gradient_steps*utd_ratio，默认：1）",
    )
    parser.add_argument(
        "--policy_delay",
        "--policy-delay",
        type=int,
        default=1,
        help="延迟策略更新频率：每多少次critic更新执行1次actor更新（默认：1）",
    )
    parser.add_argument(
        "--critic_loss",
        "--critic-loss",
        type=str,
        choices=["mse", "huber"],
        default="mse",
        help="Critic损失类型（mse/huber，默认：mse）",
    )
    parser.add_argument(
        "--grad_clip_norm",
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="梯度裁剪阈值（<=0表示关闭，默认：0）",
    )
    parser.add_argument(
        "--target_q_clip",
        "--target-q-clip",
        type=float,
        default=0.0,
        help="Target Q裁剪阈值（<=0表示关闭，默认：0）",
    )
    parser.add_argument(
        "--use_constraint_weighting",
        "--use-constraint-weighting",
        type=int,
        choices=[0, 1],
        default=0,
        help="是否启用约束感知critic loss加权（0/1，默认：0）",
    )
    parser.add_argument(
        "--action-noise-type",
        type=str,
        default="ou",
        choices=["ou", "normal", "none"],
        help="探索噪声类型（ou/normal/none），默认：ou",
    )
    parser.add_argument(
        "--action-noise-sigma",
        type=float,
        default=0.2,
        help="探索噪声标准差，默认：0.2",
    )
    parser.add_argument(
        "--action-noise-theta",
        type=float,
        default=0.15,
        help="OU噪声theta参数，默认：0.15",
    )
    parser.add_argument(
        "--use_attention",
        "--use-attention",
        type=int,
        choices=[0, 1],
        default=0,
        help="是否启用特征注意力（0/1，默认：0）",
    )
    parser.add_argument(
        "--attention_hidden_dim",
        "--attention-hidden-dim",
        type=int,
        default=64,
        help="Attention隐藏层维度（默认：64）",
    )
    parser.add_argument(
        "--attention_use_layer_norm",
        "--attention-use-layer-norm",
        type=int,
        choices=[0, 1],
        default=0,
        help="Attention输出后是否使用LayerNorm（0/1，默认：0）",
    )
    parser.add_argument(
        "--use_per",
        "--use-per",
        type=int,
        choices=[0, 1],
        default=0,
        help="是否启用PER（0/1，默认：0）",
    )
    parser.add_argument(
        "--per_alpha",
        "--per-alpha",
        type=float,
        default=0.6,
        help="PER alpha（默认：0.6）",
    )
    parser.add_argument(
        "--per_beta0",
        "--per-beta0",
        "--per_beta_start",
        "--per-beta-start",
        dest="per_beta0",
        type=float,
        default=0.4,
        help="PER beta起始值（默认：0.4）",
    )
    parser.add_argument(
        "--per_beta_end",
        "--per-beta-end",
        type=float,
        default=1.0,
        help="PER beta结束值（默认：1.0）",
    )
    parser.add_argument(
        "--per_eps",
        "--per-eps",
        type=float,
        default=1e-6,
        help="PER epsilon（默认：1e-6）",
    )
    parser.add_argument(
        "--per_clip_max",
        "--per-clip-max",
        type=float,
        default=0.0,
        help="PER priority裁剪上限（<=0 表示关闭，默认：0）",
    )
    parser.add_argument(
        "--per_mix_uniform_ratio",
        "--per-mix-uniform-ratio",
        type=float,
        default=0.0,
        help="PER混合均匀采样比例（0~1，默认：0）",
    )
    parser.add_argument(
        "--per_constraint_priority",
        "--per-constraint-priority",
        type=int,
        choices=[0, 1],
        default=0,
        help="PER是否启用约束调制priority（0/1，默认：0）",
    )
    parser.add_argument(
        "--per_constraint_scale",
        "--per-constraint-scale",
        type=float,
        default=1.0,
        help="PER约束调制缩放系数（默认：1.0）",
    )
    parser.add_argument(
        "--per_beta_anneal_steps",
        "--per-beta-anneal-steps",
        type=int,
        default=0,
        help="PER beta退火步数（默认：0，表示使用total-timesteps）",
    )
    parser.add_argument(
        "--use_nstep",
        "--use-nstep",
        type=int,
        choices=[0, 1],
        default=0,
        help="是否启用N-step return（0/1，默认：0）",
    )
    parser.add_argument(
        "--n_step",
        "--n-step",
        type=int,
        default=5,
        help="N-step长度（默认：5）",
    )
    parser.add_argument(
        "--n_step_adaptive",
        "--n-step-adaptive",
        type=int,
        choices=[0, 1],
        default=0,
        help="是否启用自适应N-step（高done比例时降半，0/1，默认：0）",
    )
    parser.add_argument(
        "--constraint_mode",
        "--constraint-mode",
        type=str,
        default="none",
        choices=["none", "lagrangian_hinge"],
        help="时延约束模式（none/lagrangian_hinge，默认：none）",
    )
    parser.add_argument(
        "--latency_limit_ms",
        "--latency-limit-ms",
        type=float,
        default=80.0,
        help="时延阈值（ms，默认：80.0）",
    )
    parser.add_argument(
        "--lambda_lr",
        "--lambda-lr",
        type=float,
        default=0.01,
        help="Lagrangian λ更新学习率（默认：0.01）",
    )
    parser.add_argument(
        "--penalty_scale",
        "--penalty-scale",
        type=float,
        default=1.0,
        help="约束惩罚缩放系数（默认：1.0）",
    )
    parser.add_argument(
        "--constraint_lambda_init",
        type=float,
        default=1.0,
        help="约束λ初始值（默认：1.0）",
    )
    parser.add_argument(
        "--constraint_lambda_max",
        type=float,
        default=100.0,
        help="约束λ上限（默认：100.0）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认：42",
    )
    # 工作负载相关参数
    parser.add_argument(
        "--enable-workload",
        action="store_true",
        help="启用工作负载（使用 emqtt_bench 生成 MQTT 消息流量，必需）",
    )
    parser.add_argument(
        "--workload-publishers",
        type=int,
        default=100,
        help="工作负载发布者数量（默认：100）",
    )
    parser.add_argument(
        "--workload-subscribers",
        type=int,
        default=10,
        help="工作负载订阅者数量（默认：10）",
    )
    parser.add_argument(
        "--workload-topic",
        type=str,
        default="test/topic",
        help="工作负载 MQTT 主题（默认：test/topic）",
    )
    parser.add_argument(
        "--workload-message-rate",
        type=int,
        default=None,
        help="工作负载消息速率（所有发布者总计的每秒消息数，默认：根据发布者间隔自动计算）",
    )
    parser.add_argument(
        "--workload-publisher-interval-ms",
        type=int,
        default=15,
        help="每个发布者发布消息的间隔（毫秒，默认：15ms，即约66.67 msg/s per publisher）",
    )
    parser.add_argument(
        "--workload-message-size",
        type=int,
        default=512,
        help="工作负载消息大小（字节，默认：512）",
    )
    parser.add_argument(
        "--workload-qos",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="工作负载 QoS 级别（默认：1）",
    )
    parser.add_argument(
        "--emqtt-bench-path",
        type=str,
        default=None,
        help="emqtt_bench 可执行文件路径（默认：从环境变量或 PATH 查找）",
    )
    # 磁盘空间优化参数
    parser.add_argument(
        "--save-replay-buffer",
        action="store_true",
        default=False,
        help="是否保存replay buffer（默认：False，不保存以节省磁盘空间）",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=3,
        help="最多保留多少个checkpoint文件（默认：3，超出会自动删除最旧的）",
    )
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        default=False,
        help="禁用TensorBoard日志以节省磁盘空间（默认：False，启用TensorBoard）",
    )
    parser.add_argument(
        "--limit-action-log",
        action="store_true",
        default=False,
        help="限制action日志大小，只记录每N步（默认：False，记录所有步）",
    )
    parser.add_argument(
        "--action-log-interval",
        type=int,
        default=10,
        help="如果启用limit-action-log，每隔多少步记录一次（默认：10）",
    )
    parser.add_argument(
        "--cleanup-mosquitto-logs",
        action="store_true",
        default=False,
        help="定期清理Mosquitto日志文件（默认：False，不清理）",
    )
    parser.add_argument(
        "--mosquitto-log-cleanup-freq",
        type=int,
        default=5000,
        help="每隔多少步清理一次Mosquitto日志（默认：5000）",
    )
    parser.add_argument(
        "--max-mosquitto-log-files",
        type=int,
        default=3,
        help="最多保留多少个Mosquitto日志文件（默认：3）",
    )
    parser.add_argument(
        "--replay-debug-freq",
        type=int,
        default=100,
        help="Replay调试指标记录频率（步，默认：100）",
    )
    return parser.parse_args()


class WorkloadHealthCheckCallback(BaseCallback):
    """
    工作负载健康检查 Callback
    定期检查工作负载是否还在运行，如果停止则尝试重启
    
    改进：
    1. 每步都检查（check_freq=1），确保Broker重启后立即恢复
    2. 在Broker重启后立即检查并重启工作负载
    3. 添加详细的状态日志
    """
    def __init__(self, workload, check_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.workload = workload
        self.check_freq = check_freq  # 检查频率（步数），默认每步检查
        self.last_check = -1  # 初始化为-1，确保第一步总是检查
        self.restart_count = 0
        self.last_broker_restart_step = -1  # 记录最后一次Broker重启的步数
        self.workload_started = False  # 标记工作负载是否已启动
    
    def _on_training_start(self) -> None:
        """训练开始时，确保工作负载已启动"""
        print("\n[工作负载健康检查] 训练开始，检查工作负载状态...")
        if not self.workload.is_running():
            print("[工作负载健康检查] 工作负载未运行，尝试启动...")
            try:
                if self.workload._last_config is not None:
                    self.workload.restart()
                else:
                    print("[工作负载健康检查] ⚠️  没有保存的配置，无法重启工作负载")
                    print("[工作负载健康检查] 请确保训练脚本使用--enable-workload参数")
            except Exception as e:
                print(f"[工作负载健康检查] ❌ 启动失败: {e}")
        else:
            print("[工作负载健康检查] ✅ 工作负载已运行")
            self.workload_started = True
    
    def _on_step(self) -> bool:
        """每步检查工作负载健康状态"""
        # 每步都检查（check_freq=1），确保Broker重启后立即恢复
        should_check = (
            self.num_timesteps - self.last_check >= self.check_freq or
            self.num_timesteps == 0  # 第一步总是检查
        )
        
        if should_check:
            self.last_check = self.num_timesteps
            
            # 检查Broker是否重启（通过检查环境的_broker_restart_steps和_need_workload_restart属性）
            # 如果Broker重启，立即重启工作负载
            broker_restarted = False
            try:
                # 尝试从环境中获取Broker重启信息
                env = self.training_env
                if hasattr(env, 'envs'):
                    # 如果是向量化环境，取第一个环境
                    env = env.envs[0]
                if hasattr(env, 'unwrapped'):
                    env = env.unwrapped
                if hasattr(env, 'env'):
                    env = env.env
                if hasattr(env, 'unwrapped'):
                    env = env.unwrapped
                
                # 优先检查_need_workload_restart标志（最直接的方式）
                if hasattr(env, '_need_workload_restart') and env._need_workload_restart:
                    broker_restarted = True
                    print(f"\n[工作负载健康检查] 🔄 检测到Broker重启标志，立即重启工作负载（步数: {self.num_timesteps}）...")
                # 如果没有标志，检查_broker_restart_steps（向后兼容）
                elif hasattr(env, '_broker_restart_steps'):
                    if len(env._broker_restart_steps) > 0:
                        last_restart_step = env._broker_restart_steps[-1]
                        # 如果Broker在最近几步重启，标记需要重启工作负载
                        if self.num_timesteps - last_restart_step <= 2:
                            broker_restarted = True
                            print(f"\n[工作负载健康检查] 🔄 检测到Broker在步数 {last_restart_step} 重启，立即重启工作负载...")
            except Exception as e:
                # 如果无法获取Broker重启信息，忽略错误
                pass
            
            # 检查工作负载是否运行
            if not self.workload.is_running() or broker_restarted:
                if broker_restarted:
                    print(f"[工作负载健康检查] Broker重启导致工作负载断开，立即重启...")
                else:
                    self.restart_count += 1
                    print(f"\n[工作负载健康检查] ⚠️  工作负载在步数 {self.num_timesteps} 时停止运行")
                    print(f"[工作负载健康检查] 尝试重启工作负载（第 {self.restart_count} 次）...")
                
                try:
                    if self.workload._last_config is not None:
                        # 立即重启工作负载（使用保存的配置）
                        print(f"[工作负载健康检查] 正在重启工作负载（使用原配置：{self.workload._last_config.num_publishers}发布者，{self.workload._last_config.num_subscribers}订阅者，主题'{self.workload._last_config.topic}'，QoS={self.workload._last_config.qos}，间隔={self.workload._last_config.publisher_interval_ms}ms，消息大小={self.workload._last_config.message_size}B）...")
                        self.workload.restart()
                        print(f"[工作负载健康检查] ✅ 工作负载重启成功，等待稳定运行（30秒）...")
                        import time
                        time.sleep(30.0)  # 等待工作负载稳定运行30秒
                        # 再次验证工作负载是否运行
                        if self.workload.is_running():
                            print(f"[工作负载健康检查] ✅ 工作负载已稳定运行（进程数: {len(self.workload._processes)}）")
                            # 验证工作负载是否真的在发送消息
                            if self.workload._last_config.num_publishers > 0:
                                print(f"[工作负载健康检查] 验证工作负载消息发送（订阅主题 '{self.workload._last_config.topic}' 等待5秒）...")
                                if self.workload._verify_messages_sending(self.workload._last_config.topic, timeout_sec=5.0):
                                    print(f"[工作负载健康检查] ✅ 验证成功：工作负载正在发送消息到主题 '{self.workload._last_config.topic}'")
                                    print(f"[工作负载健康检查] 提示：可以使用以下命令监听消息:")
                                    print(f"  mosquitto_sub -h {self.workload.broker_host} -p {self.workload.broker_port} -t '{self.workload._last_config.topic}' -v")
                                else:
                                    print(f"[工作负载健康检查] ⚠️  警告：无法验证消息发送，但进程仍在运行")
                                    print(f"[工作负载健康检查] 提示：可以使用以下命令手动验证:")
                                    print(f"  mosquitto_sub -h {self.workload.broker_host} -p {self.workload.broker_port} -t '{self.workload._last_config.topic}' -C 1")
                            self.workload_started = True
                            # 清除Broker重启标志（如果存在）
                            try:
                                env = self.training_env
                                if hasattr(env, 'envs'):
                                    env = env.envs[0]
                                if hasattr(env, 'unwrapped'):
                                    env = env.unwrapped
                                if hasattr(env, 'env'):
                                    env = env.env
                                if hasattr(env, 'unwrapped'):
                                    env = env.unwrapped
                                if hasattr(env, '_need_workload_restart'):
                                    env._need_workload_restart = False
                            except:
                                pass
                        else:
                            print(f"[工作负载健康检查] ⚠️  工作负载重启后仍未运行，将在下一步继续检查")
                    else:
                        print("[工作负载健康检查] ❌ 无法重启：没有保存的配置")
                        print("[工作负载健康检查] 请确保训练脚本使用--enable-workload参数")
                except Exception as e:
                    print(f"[工作负载健康检查] ❌ 重启失败: {e}")
                    import traceback
                    traceback.print_exc()
                    print("[工作负载健康检查] 训练将继续，但可能无法获得有效的奖励信号")
            else:
                # 工作负载正在运行
                if not self.workload_started:
                    print(f"[工作负载健康检查] ✅ 工作负载运行正常（步数: {self.num_timesteps}）")
                    self.workload_started = True
                # 每50步打印一次状态（减少日志）
                elif self.num_timesteps % 50 == 0:
                    print(f"[工作负载健康检查] ✅ 工作负载运行正常（步数: {self.num_timesteps}，重启次数: {self.restart_count}）")
        
        return True


class ActionThroughputLoggerWrapper(gym.Env):
    """
    包装环境，记录每一步的action和吞吐量
    将数据保存到CSV文件中
    
    继承自gym.Env以确保与Monitor兼容
    """
    def __init__(self, env, save_path: str, log_interval: int = 1):
        super().__init__()
        self.env = env
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV文件路径
        self.csv_path = self.save_path / "action_throughput_log.csv"
        
        # 日志记录间隔（每N步记录一次，1表示每步都记录）
        self.log_interval = log_interval
        
        # 当前episode编号和步数
        self.current_episode = 0
        self.current_step = 0
        
        # 缓存 knob_space 引用（在 step 中惰性初始化）
        self._cached_knob_space = None
        
        # 动作名称（11维）- 归一化的action值
        self.action_names = [
            "action_0_max_inflight_messages",
            "action_1_max_inflight_bytes",
            "action_2_max_queued_messages",
            "action_3_max_queued_bytes",
            "action_4_queue_qos0_messages",
            "action_5_memory_limit",
            "action_6_persistence",
            "action_7_autosave_interval",
            "action_8_set_tcp_nodelay",
            "action_9_max_packet_size",
            "action_10_message_size_limit",
        ]
        
        # 解码后的配置参数名称
        self.knob_names = [
            "decoded_max_inflight_messages",
            "decoded_max_inflight_bytes",
            "decoded_max_queued_messages",
            "decoded_max_queued_bytes",
            "decoded_queue_qos0_messages",
            "decoded_memory_limit",
            "decoded_persistence",
            "decoded_autosave_interval",
            "decoded_set_tcp_nodelay",
            "decoded_max_packet_size",
            "decoded_message_size_limit",
        ]

        # $SYS 关键指标（用于诊断吞吐来源）
        self.sys_metric_names = [
            "sys_clients_connected",
            "sys_msgs_received_1min",
            "sys_msgs_received_1min_per_sec",
            "sys_msgs_received_total",
            "sys_msgs_received_rate",
        ]
        self.sys_metric_keys = [
            "$SYS/broker/clients/connected",
            "$SYS/broker/load/messages/received/1min",
            "$SYS/broker/load/messages/received/1min_per_sec",
            "$SYS/broker/messages/received",
            "$SYS/broker/messages/received_rate",
        ]
        
        # 初始化CSV文件，写入表头
        self._init_csv()
        
        # 代理action_space和observation_space属性
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, 'metadata', {})
    
    def _init_csv(self):
        """初始化CSV文件，写入表头（每次训练开始时覆盖旧文件）"""
        # 确保目录存在且权限正确
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 每次训练开始时，覆盖旧文件（使用'w'模式）
        try:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # 表头：步数、episode、11个action值（归一化）、11个解码后的配置值、吞吐量、奖励
                header = (
                    ["step", "episode"] +
                    self.action_names +
                    self.knob_names +
                    self.sys_metric_names +
                    [
                        "throughput",
                        "throughput_msg_per_sec",
                        "latency_p50_ms",
                        "latency_p95_ms",
                        "queue_depth_norm",
                        "reward",
                        "restart_count",
                        "consecutive_failures",
                        "reward_throughput_base",
                        "reward_throughput_step",
                        "reward_latency_base",
                        "reward_latency_step",
                        "latency_source",
                        "latency_probe_connected",
                        "latency_probe_samples",
                        "latency_probe_min",
                        "latency_probe_max",
                        "constraint_lambda",
                        "constraint_penalty",
                        "latency_limit_ms",
                        "latency_violation_ms",
                        "constraint_metric_ms",
                        "unsafe",
                    ]
                )
                # 注意：未来可以添加更多状态指标到CSV，如延迟等
                writer.writerow(header)
                f.flush()  # 确保立即写入磁盘
                import os
                os.fsync(f.fileno())  # 强制同步到磁盘
            print(f"[ActionThroughputLogger] ✅ CSV文件已初始化（覆盖模式）: {self.csv_path}")
            print(f"[ActionThroughputLogger] CSV包含: action值（归一化）+ 解码后的配置值 + 吞吐量 + 奖励")
            print(f"[ActionThroughputLogger] 注意: 状态空间已扩展到10维，包含延迟和历史信息")
        except PermissionError as e:
            print(f"[ActionThroughputLogger] ❌ 无法创建CSV文件（权限不足）: {e}")
            print(f"[ActionThroughputLogger] 文件路径: {self.csv_path}")
            print(f"[ActionThroughputLogger] 提示: 请确保目录可写，或使用 sudo chown 修改权限")
        except Exception as e:
            print(f"[ActionThroughputLogger] ❌ 初始化CSV文件失败: {e}")
    
    def reset(self, **kwargs):
        """重置环境，开始新episode"""
        self.current_episode += 1
        self.current_step = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """执行一步，记录action和吞吐量"""
        self.current_step += 1
        
        # 执行环境step
        if self.current_step <= 3 or self.current_step % 20 == 0:
            print(f"[ActionThroughputLogger] 执行env.step()（步数: {self.current_step}）...")
        result = self.env.step(action)
        
        # 根据log_interval决定是否记录日志
        # 默认log_interval=1，表示每步都记录
        should_log = (self.current_step % self.log_interval == 0) or (self.current_step <= 3)
        
        # 每步都记录（如果log_interval=1）
        if self.log_interval == 1 and self.current_step % 100 == 0:
            print(f"[ActionThroughputLogger] 已记录 {self.current_step} 步数据到CSV（episode {self.current_episode}）")
        
        if self.current_step <= 3 or self.current_step % 20 == 0:
            print(f"[ActionThroughputLogger] env.step() 完成，解析返回值...")
        
        # 解析返回值（兼容gymnasium的5元组格式）
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated

        # 验证状态向量维度（扩展后应为10维）
        if len(obs) != 10:
            print(f"[ActionThroughputLogger] ⚠️  警告: 状态向量维度为{len(obs)}，期望10维")
        
        if self.current_step <= 3 or self.current_step % 20 == 0:
            print(f"[ActionThroughputLogger] 返回值解析完成: reward={reward:.6f}, terminated={terminated}, truncated={truncated}")
        
        # 提取吞吐量（从状态向量的第1维，即消息速率归一化值）
        # state[1] 是 msg_rate_norm，表示消息速率（吞吐量的代理指标）
        # 注意：状态空间已扩展到10维，第1维仍然是吞吐量
        throughput = float(obs[1]) if len(obs) > 1 else 0.0
        
        if self.current_step <= 3 or self.current_step % 20 == 0:
            print(f"[ActionThroughputLogger] 吞吐量提取完成: {throughput:.10f}")
            # 显示其他关键指标（如果状态向量足够长）
            if len(obs) >= 6:
                latency_p50 = float(obs[5])
                print(f"[ActionThroughputLogger] P50延迟: {latency_p50:.10f}")
            if len(obs) >= 10:
                throughput_avg = float(obs[8])
                latency_avg = float(obs[9])
                print(f"[ActionThroughputLogger] 历史平均 - 吞吐量: {throughput_avg:.10f}, 延迟: {latency_avg:.10f}")
        
        # 解码action为实际配置值
        if self.current_step <= 3 or self.current_step % 20 == 0:
            print(f"[ActionThroughputLogger] 开始解码action...")
        
        decoded_values = ["unlimited", "unlimited", "unlimited", "unlimited", "False", 
                         "unlimited", "False", "1800", "False", "unlimited", "unlimited"]  # 默认值
        try:
            # 获取knob_space（可能被Monitor包装，需要unwrapped）
            # 使用缓存避免每次都查找，并在初始化时保存knob_space引用
            if self._cached_knob_space is None:
                if self.current_step <= 3:
                    print(f"[ActionThroughputLogger] 首次查找knob_space...")
                env_with_knob_space = self.env
                max_unwrap_depth = 10  # 防止无限循环
                unwrap_count = 0
                last_env = None
                while unwrap_count < max_unwrap_depth:
                    if env_with_knob_space is last_env:
                        # 防止循环引用
                        break
                    last_env = env_with_knob_space
                    
                    if hasattr(env_with_knob_space, 'knob_space'):
                        self._cached_knob_space = env_with_knob_space.knob_space
                        if self.current_step <= 3:
                            print(f"[ActionThroughputLogger] ✅ 找到knob_space（深度: {unwrap_count}）")
                        break
                    elif hasattr(env_with_knob_space, 'unwrapped'):
                        env_with_knob_space = env_with_knob_space.unwrapped
                        unwrap_count += 1
                    elif hasattr(env_with_knob_space, 'env'):
                        env_with_knob_space = env_with_knob_space.env
                        unwrap_count += 1
                    else:
                        break
                else:
                    # 如果循环结束还没找到
                    if self.current_step <= 3:
                        print(f"[ActionThroughputLogger] ⚠️  未找到knob_space（已搜索深度: {unwrap_count}）")
                    self._cached_knob_space = None
            
            # 使用缓存的knob_space
            if hasattr(self, '_cached_knob_space') and self._cached_knob_space is not None:
                if self.current_step <= 3 or self.current_step % 20 == 0:
                    print(f"[ActionThroughputLogger] 使用缓存的knob_space，开始解码...")
                knobs = self._cached_knob_space.decode_action(action)
                if self.current_step <= 3 or self.current_step % 20 == 0:
                    print(f"[ActionThroughputLogger] decode_action完成，提取值...")
                # 按照knob_names的顺序提取解码后的值
                # 对于0值（表示unlimited），显示为"unlimited"字符串
                # 对于布尔值，显示为"True"/"False"字符串
                def format_value(key: str, value):
                    """格式化配置值：0显示为unlimited，布尔值显示为True/False"""
                    if key in ["queue_qos0_messages", "persistence", "set_tcp_nodelay"]:
                        # 布尔值
                        return "True" if value else "False"
                    elif key in ["max_inflight_bytes", "max_queued_bytes", "memory_limit", 
                                 "max_packet_size", "message_size_limit"]:
                        # 这些配置项的0值表示unlimited
                        return "unlimited" if value == 0 else str(value)
                    else:
                        # 其他配置项：0值也显示为unlimited（对于max_inflight_messages和max_queued_messages）
                        if key in ["max_inflight_messages", "max_queued_messages"] and value == 0:
                            return "unlimited"
                        return str(value)
                
                decoded_values = [
                    format_value("max_inflight_messages", knobs.get("max_inflight_messages", 0)),
                    format_value("max_inflight_bytes", knobs.get("max_inflight_bytes", 0)),
                    format_value("max_queued_messages", knobs.get("max_queued_messages", 0)),
                    format_value("max_queued_bytes", knobs.get("max_queued_bytes", 0)),
                    format_value("queue_qos0_messages", knobs.get("queue_qos0_messages", False)),
                    format_value("memory_limit", knobs.get("memory_limit", 0)),
                    format_value("persistence", knobs.get("persistence", False)),
                    format_value("autosave_interval", knobs.get("autosave_interval", 0)),
                    format_value("set_tcp_nodelay", knobs.get("set_tcp_nodelay", False)),
                    format_value("max_packet_size", knobs.get("max_packet_size", 0)),
                    format_value("message_size_limit", knobs.get("message_size_limit", 0)),
                ]
                if self.current_step <= 3 or self.current_step % 20 == 0:
                    print(f"[ActionThroughputLogger] action解码完成: max_inflight_messages={decoded_values[0]}")
            else:
                # 如果没有knob_space，使用默认值填充
                if self.current_step <= 3 or self.current_step % 20 == 0:
                    print(f"[ActionThroughputLogger] ⚠️  未找到knob_space，使用默认值填充")
                decoded_values = ["unlimited", "unlimited", "unlimited", "unlimited", "False", 
                                 "unlimited", "False", "1800", "False", "unlimited", "unlimited"]
        except Exception as e:
            print(f"[ActionThroughputLogger] ❌ 解码action失败: {e}")
            import traceback
            traceback.print_exc()
            decoded_values = ["unlimited", "unlimited", "unlimited", "unlimited", "False", 
                             "unlimited", "False", "1800", "False", "unlimited", "unlimited"]  # 如果解码失败，使用默认值填充

        # 读取最近一次 broker $SYS 指标
        sys_values = []
        try:
            metrics_env = self.env
            max_unwrap_depth = 10
            unwrap_count = 0
            last_env = None
            while unwrap_count < max_unwrap_depth:
                if metrics_env is last_env:
                    break
                last_env = metrics_env
                if hasattr(metrics_env, "get_last_broker_metrics"):
                    break
                if hasattr(metrics_env, "unwrapped"):
                    metrics_env = metrics_env.unwrapped
                    unwrap_count += 1
                elif hasattr(metrics_env, "env"):
                    metrics_env = metrics_env.env
                    unwrap_count += 1
                else:
                    break
            if hasattr(metrics_env, "get_last_broker_metrics"):
                metrics = metrics_env.get_last_broker_metrics()
            elif hasattr(metrics_env, "_last_broker_metrics"):
                metrics = metrics_env._last_broker_metrics
            else:
                metrics = {}
        except Exception:
            metrics = {}

        for key in self.sys_metric_keys:
            value = metrics.get(key)
            sys_values.append(value if value is not None else "")
        
        # 记录到CSV文件（根据log_interval决定是否记录）
        if should_log:
            if self.current_step <= 3 or self.current_step % 20 == 0:
                print(f"[ActionThroughputLogger] 开始写入CSV文件...")
            try:
                latency_source = info.get("latency_source", "")
                latency_probe_connected = info.get("latency_probe_connected", "")
                latency_probe_samples = info.get("latency_probe_samples", "")
                latency_probe_min = info.get("latency_probe_min", "")
                latency_probe_max = info.get("latency_probe_max", "")
                throughput_msg_per_sec = info.get("throughput_msg_per_sec", throughput * 10000.0)
                latency_p50_ms = info.get("latency_p50_ms", float(obs[5]) * 100.0 if len(obs) > 5 else 0.0)
                latency_p95_ms = info.get("latency_p95_ms", float(obs[6]) * 100.0 if len(obs) > 6 else 0.0)
                queue_depth_norm = info.get("queue_depth_norm", float(obs[7]) if len(obs) > 7 else 0.0)
                restart_count = info.get("restart_count", "")
                consecutive_failures = info.get("consecutive_failures", "")
                reward_components = info.get("reward_components", {}) or {}
                reward_tp_base = reward_components.get("throughput_base", "")
                reward_tp_step = reward_components.get("throughput_step", "")
                reward_lat_base = reward_components.get("latency_base", "")
                reward_lat_step = reward_components.get("latency_step", "")
                constraint_lambda = reward_components.get("constraint_lambda", "")
                constraint_penalty = reward_components.get("constraint_penalty", "")
                latency_limit_ms = reward_components.get("latency_limit_ms", "")
                latency_violation_ms = reward_components.get("latency_violation_ms", "")
                constraint_metric_ms = reward_components.get("constraint_metric_ms", "")
                unsafe = info.get("unsafe", reward_components.get("unsafe", ""))
                # 将action转换为列表（如果是numpy数组）
                action_list = action.tolist() if hasattr(action, 'tolist') else list(action)
                # 行数据：步数、episode、11个action值（归一化）、11个解码后的配置值、吞吐量、奖励
                # 注意：扩展状态向量后，可以添加更多指标到CSV
                row = (
                    [self.current_step, self.current_episode] +
                    action_list +
                    decoded_values +
                    sys_values +
                    [
                        throughput,
                        throughput_msg_per_sec,
                        latency_p50_ms,
                        latency_p95_ms,
                        queue_depth_norm,
                        reward,
                        restart_count,
                        consecutive_failures,
                        reward_tp_base,
                        reward_tp_step,
                        reward_lat_base,
                        reward_lat_step,
                        latency_source,
                        latency_probe_connected,
                        latency_probe_samples,
                        latency_probe_min,
                        latency_probe_max,
                        constraint_lambda,
                        constraint_penalty,
                        latency_limit_ms,
                        latency_violation_ms,
                        constraint_metric_ms,
                        unsafe,
                    ]
                )
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                    f.flush()  # 确保立即写入磁盘
                    import os
                    os.fsync(f.fileno())  # 强制同步到磁盘
                if self.current_step <= 3 or self.current_step % 20 == 0:
                    print(f"[ActionThroughputLogger] CSV写入完成（步数: {self.current_step}, episode: {self.current_episode}）")
            except PermissionError as e:
                # 如果权限不足，打印详细错误信息
                import os
                import stat
                try:
                    file_stat = self.csv_path.stat()
                    file_owner = f"uid={file_stat.st_uid}, gid={file_stat.st_gid}"
                    current_uid = os.getuid()
                    current_gid = os.getgid()
                    print(f"[ActionThroughputLogger] ❌ 权限不足，无法写入CSV文件")
                    print(f"[ActionThroughputLogger] 文件路径: {self.csv_path}")
                    print(f"[ActionThroughputLogger] 文件所有者: {file_owner}")
                    print(f"[ActionThroughputLogger] 当前用户: uid={current_uid}, gid={current_gid}")
                    print(f"[ActionThroughputLogger] 提示: 请使用以下命令修复权限:")
                    print(f"[ActionThroughputLogger]   sudo chown {os.getenv('USER', 'qincai')}:{os.getenv('USER', 'qincai')} {self.csv_path}")
                except Exception as e2:
                    print(f"[ActionThroughputLogger] ⚠️  无法写入CSV文件: {e}")
                    print(f"[ActionThroughputLogger] 文件路径: {self.csv_path}")
            except Exception as e:
                print(f"[ActionThroughputLogger] ⚠️  写入CSV文件时出错: {e}")
                print(f"[ActionThroughputLogger] 文件路径: {self.csv_path}")
                import traceback
                traceback.print_exc()
        
        # 返回原始结果
        if len(result) == 4:
            return obs, reward, done, info
        else:
            return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """代理其他属性和方法到原始环境"""
        return getattr(self.env, name)
    
    def close(self):
        """关闭环境"""
        print(f"\n[ActionThroughputLogger] 已记录 {self.current_step} 步数据（episode {self.current_episode}）")
        print(f"[ActionThroughputLogger] 数据已保存到: {self.csv_path}")
        return self.env.close()


class CheckpointCleanupCallback(BaseCallback):
    """
    定期清理旧的checkpoint文件，只保留最新的N个
    """
    def __init__(self, save_dir: Path, max_checkpoints: int = 3, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.check_freq = check_freq
        self.last_cleanup = -1
    
    def _on_step(self) -> bool:
        """定期清理旧的checkpoint"""
        if self.num_timesteps - self.last_cleanup >= self.check_freq:
            self.last_cleanup = self.num_timesteps
            self._cleanup_old_checkpoints()
        return True
    
    def _cleanup_old_checkpoints(self):
        """删除旧的checkpoint文件，只保留最新的N个"""
        try:
            # 查找所有checkpoint zip文件
            checkpoint_files = sorted(
                self.save_dir.glob("ddpg_mosquitto_*_steps.zip"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # 如果超过最大数量，删除最旧的
            if len(checkpoint_files) > self.max_checkpoints:
                files_to_delete = checkpoint_files[self.max_checkpoints:]
                for file in files_to_delete:
                    # 同时删除对应的replay buffer文件
                    replay_buffer_file = file.parent / file.name.replace(".zip", "_replay_buffer.pkl")
                    if replay_buffer_file.exists():
                        replay_buffer_file.unlink()
                        if self.verbose > 0:
                            print(f"[Checkpoint清理] 删除旧的replay buffer: {replay_buffer_file.name}")
                    
                    file.unlink()
                    if self.verbose > 0:
                        print(f"[Checkpoint清理] 删除旧的checkpoint: {file.name} (保留最新的{self.max_checkpoints}个)")
        except Exception as e:
            if self.verbose > 0:
                print(f"[Checkpoint清理] 清理时出错: {e}")


class MosquittoLogCleanupCallback(BaseCallback):
    """
    定期清理Mosquitto日志文件，防止磁盘空间被占满
    """
    def __init__(self, log_dir: str = "/var/log/mosquitto", check_freq: int = 5000, max_log_files: int = 3, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.check_freq = check_freq
        self.max_log_files = max_log_files
        self.last_cleanup = -1
    
    def _on_step(self) -> bool:
        """定期清理Mosquitto日志"""
        if self.num_timesteps - self.last_cleanup >= self.check_freq:
            self.last_cleanup = self.num_timesteps
            self._cleanup_mosquitto_logs()
        return True
    
    def _cleanup_mosquitto_logs(self):
        """清理Mosquitto日志文件"""
        try:
            if not self.log_dir.exists():
                return
            
            # 清理旧的压缩日志文件（只保留最新的N个）
            import subprocess
            gz_files = sorted(
                self.log_dir.glob("*.log.*.gz"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            if len(gz_files) > self.max_log_files:
                files_to_delete = gz_files[self.max_log_files:]
                for file in files_to_delete:
                    try:
                        # 使用sudo删除（需要root权限）
                        subprocess.run(
                            ["sudo", "rm", "-f", str(file)],
                            check=False,
                            capture_output=True
                        )
                        if self.verbose > 0:
                            print(f"[Mosquitto日志清理] 删除旧日志: {file.name}")
                    except Exception:
                        pass  # 忽略删除失败（可能没有权限）
            
            # 检查当前日志文件大小，如果超过100MB则清空
            current_log = self.log_dir / "mosquitto.log"
            if current_log.exists():
                size_mb = current_log.stat().st_size / (1024 * 1024)
                if size_mb > 100:
                    try:
                        subprocess.run(
                            ["sudo", "truncate", "-s", "0", str(current_log)],
                            check=False,
                            capture_output=True
                        )
                        if self.verbose > 0:
                            print(f"[Mosquitto日志清理] 清空当前日志文件（大小: {size_mb:.1f}MB）")
                    except Exception:
                        pass  # 忽略清空失败（可能没有权限）
        except Exception as e:
            if self.verbose > 0:
                print(f"[Mosquitto日志清理] 清理时出错: {e}")


class ProgressBarCallback(BaseCallback):
    """
    显示训练进度条的 Callback
    """
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_timesteps = 0
        
    def _on_training_start(self) -> None:
        """训练开始时创建进度条"""
        if TQDM_AVAILABLE:
            self.pbar = tqdm(
                total=self.total_timesteps,
                desc="训练进度",
                unit="step",
                unit_scale=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
        else:
            print(f"开始训练，总步数: {self.total_timesteps}")
    
    def _on_step(self) -> bool:
        """每步更新进度条"""
        if self.pbar is not None:
            # 计算新增的步数（因为 num_timesteps 是按 rollout 更新的）
            new_timesteps = self.num_timesteps - self.last_timesteps
            if new_timesteps > 0:
                self.pbar.update(new_timesteps)
                self.last_timesteps = self.num_timesteps
                # 更新进度条描述，显示当前步数和总步数
                progress_pct = (self.num_timesteps / self.total_timesteps) * 100
                self.pbar.set_description(
                    f"训练进度 [{progress_pct:.1f}%]"
                )
        return True
    
    def _on_training_end(self) -> None:
        """训练结束时关闭进度条"""
        if self.pbar is not None:
            # 确保进度条到达100%
            remaining = self.total_timesteps - self.last_timesteps
            if remaining > 0:
                self.pbar.update(remaining)
            self.pbar.close()
            print(f"\n训练完成！总步数: {self.num_timesteps:,}")


class ReplayDebugCallback(BaseCallback):
    """
    周期性记录 replay buffer 内部统计，便于验证 PER/N-step 是否生效。
    """

    def __init__(self, check_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = max(1, int(check_freq))
        self.last_check = -1

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_check < self.check_freq:
            return True
        self.last_check = self.num_timesteps

        replay_buffer = getattr(self.model, "replay_buffer", None)
        if replay_buffer is None or not hasattr(replay_buffer, "get_debug_stats"):
            return True

        try:
            stats = replay_buffer.get_debug_stats()
        except Exception:
            return True

        for key, value in stats.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                self.logger.record(f"train/{key}", float(value))

        if self.verbose > 0:
            print(f"[ReplayDebug] step={self.num_timesteps}, stats={stats}")
        return True


def record_default_baseline(env, save_dir: Path) -> None:
    """
    在训练开始前记录默认配置下的基线性能。
    记录的是归一化后的状态向量与吞吐量估计。
    """
    print("\n" + "=" * 80)
    print("记录默认配置基线性能（默认参数 + 当前工作负载）...")
    print("=" * 80)
    try:
        result = env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        obs_list = [float(x) for x in obs]
        throughput_norm = float(obs_list[1]) if len(obs_list) > 1 else 0.0
        sys_metrics = {}
        try:
            metrics_env = env
            max_unwrap_depth = 10
            unwrap_count = 0
            last_env = None
            while unwrap_count < max_unwrap_depth:
                if metrics_env is last_env:
                    break
                last_env = metrics_env
                if hasattr(metrics_env, "get_last_broker_metrics"):
                    break
                if hasattr(metrics_env, "unwrapped"):
                    metrics_env = metrics_env.unwrapped
                    unwrap_count += 1
                elif hasattr(metrics_env, "env"):
                    metrics_env = metrics_env.env
                    unwrap_count += 1
                else:
                    break
            if hasattr(metrics_env, "get_last_broker_metrics"):
                sys_metrics = metrics_env.get_last_broker_metrics()
            elif hasattr(metrics_env, "_last_broker_metrics"):
                sys_metrics = metrics_env._last_broker_metrics
        except Exception:
            sys_metrics = {}

        baseline = {
            "timestamp": time.time(),
            "throughput_norm": throughput_norm,
            "estimated_msg_rate": throughput_norm * 10000.0,
            "state": obs_list,
            "sys_metrics": sys_metrics,
            "note": "state values are normalized; estimated_msg_rate uses 10000.0 scale",
        }

        baseline_path = save_dir / "baseline_metrics.json"
        baseline_path.write_text(
            json.dumps(baseline, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"[基线] ✅ 已记录: {baseline_path}")
        print(f"[基线] 吞吐量(归一化): {throughput_norm:.6f}")
    except Exception as e:
        print(f"[基线] ❌ 记录失败: {e}")


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    env_cfg = EnvConfig()
    env_cfg.constraint_mode = str(args.constraint_mode)
    env_cfg.latency_limit_ms = float(args.latency_limit_ms)
    env_cfg.lambda_lr = float(args.lambda_lr)
    env_cfg.penalty_scale = float(args.penalty_scale)
    env_cfg.constraint_lambda_init = float(args.constraint_lambda_init)
    env_cfg.constraint_lambda_max = float(args.constraint_lambda_max)
    
    # 创建工作负载管理器（必须启用，在创建环境之前）
    workload = None
    if not args.enable_workload:
        print("\n" + "=" * 80)
        print("错误: 训练必须在有工作负载的情况下进行！")
        print("=" * 80)
        print("\n请使用 --enable-workload 参数启用工作负载")
        print("\n示例命令:")
        print("  ./script/run_train.sh --enable-workload --total-timesteps 1000")
        print("\n工作负载配置:")
        print("  --workload-publishers 100        # 发布者数量")
        print("  --workload-subscribers 10        # 订阅者数量")
        print("  --workload-publisher-interval-ms 15  # 发布间隔（毫秒）")
        print("  --workload-message-size 512      # 消息大小（字节）")
        print("  --workload-qos 1                 # QoS 级别")
        print("=" * 80)
        sys.exit(1)
    
    # 工作负载是必需的，检查是否可用
    if not WORKLOAD_AVAILABLE:
        print("\n" + "=" * 80)
        print("错误: 无法导入 WorkloadManager，工作负载功能不可用")
        print("=" * 80)
        print("请确保 script/workload.py 文件存在")
        print("=" * 80)
        sys.exit(1)
    
    # 创建工作负载管理器（在创建环境之前，以便传递给环境）
    print("\n" + "=" * 80)
    print("创建工作负载管理器...")
    print("=" * 80)
    try:
        workload = WorkloadManager(
            broker_host=env_cfg.mqtt.host,
            broker_port=env_cfg.mqtt.port,
            emqtt_bench_path=args.emqtt_bench_path,
        )
        
        # 使用 WorkloadConfig 来精确控制发布者间隔
        from script.workload import WorkloadConfig
        workload_config = WorkloadConfig(
            num_publishers=args.workload_publishers,
            num_subscribers=args.workload_subscribers,
            topic=args.workload_topic,
            message_size=args.workload_message_size,
            qos=args.workload_qos,
            publisher_interval_ms=args.workload_publisher_interval_ms,
            duration=0,  # 持续运行直到训练结束
        )
        
        # 保存配置（用于后续重启）
        workload._last_config = workload_config
        
        print(f"[工作负载] ✅ 工作负载管理器创建成功")
        print(f"[工作负载] 配置: {args.workload_publishers}发布者，{args.workload_subscribers}订阅者")
        print(f"[工作负载] 主题: {args.workload_topic}, QoS: {args.workload_qos}")
        print(f"[工作负载] 发布者间隔: {args.workload_publisher_interval_ms}ms")
        print(f"[工作负载] 消息大小: {args.workload_message_size}B")
    except Exception as e:
        print(f"\n" + "=" * 80)
        print("错误: 创建工作负载管理器失败")
        print("=" * 80)
        print(f"错误详情: {e}")
        sys.exit(1)
    
    # 创建环境（传入工作负载管理器，以便Broker重启后自动重启工作负载）
    print("\n" + "=" * 80)
    print("创建环境...")
    print("=" * 80)
    env = make_env(env_cfg, workload_manager=workload)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(args.seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(args.seed)
    
    # 保存原始环境的配置引用（Monitor包装后会无法直接访问）
    # 注意：env 可能是 Monitor 包装后的环境，需要通过 env.unwrapped 或 env.env 访问原始环境
    original_env = env
    
    # 使用ActionThroughputLogger包装环境，记录每一步的action和吞吐量
    # 根据参数决定日志记录间隔
    log_interval = args.action_log_interval if args.limit_action_log else 1
    env = ActionThroughputLoggerWrapper(env, str(args.save_dir), log_interval=log_interval)
    if args.limit_action_log:
        print(f"[ActionThroughputLogger] 已启用日志限制：每{log_interval}步记录一次（节省磁盘空间）")
    
    # 使用Monitor包装环境，记录episode统计信息
    monitor_log_dir = Path(args.save_dir) / "monitor"
    monitor_log_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(env, str(monitor_log_dir))
    
    # 获取原始环境的配置（用于后续使用）
    # Monitor 包装后的环境可以通过 env.unwrapped 或 env.env 访问原始环境
    if hasattr(env, 'unwrapped'):
        env_with_cfg = env.unwrapped
    elif hasattr(env, 'env'):
        env_with_cfg = env.env
    else:
        env_with_cfg = original_env
    
    # 启动工作负载
    print("\n" + "=" * 80)
    print("启动工作负载（emqtt_bench）...")
    print("=" * 80)
    try:
        # 计算消息速率（用于显示）
        messages_per_publisher_per_sec = 1000.0 / args.workload_publisher_interval_ms
        total_message_rate = int(messages_per_publisher_per_sec * args.workload_publishers)
        
        # 启动工作负载
        workload.start(config=workload_config)
        print(f"[工作负载] ✅ 工作负载启动成功！")
        print(f"[工作负载] 发布者: {args.workload_publishers}, 订阅者: {args.workload_subscribers}")
        print(f"[工作负载] 主题: {args.workload_topic}, QoS: {args.workload_qos}")
        print(f"[工作负载] 发布者间隔: {args.workload_publisher_interval_ms}ms")
        print(f"[工作负载] 消息大小: {args.workload_message_size}B")
        print(f"[工作负载] 总消息速率: ~{total_message_rate} msg/s (每个发布者 ~{messages_per_publisher_per_sec:.2f} msg/s)")
        
        # 等待工作负载稳定，然后验证是否运行
        print(f"[工作负载] 等待工作负载稳定（30秒）...")
        time.sleep(30)
        
        if workload.is_running():
            print(f"[工作负载] ✅ 工作负载运行正常（进程数: {len(workload._processes)}）")
            
            # 验证工作负载是否真的在发送消息
            print(f"[工作负载] 验证消息发送（订阅主题 '{args.workload_topic}' 等待5秒）...")
            if workload._verify_messages_sending(args.workload_topic, timeout_sec=5.0):
                print(f"[工作负载] ✅ 验证成功：工作负载正在发送消息到主题 '{args.workload_topic}'")
                print(f"[工作负载] 提示：可以使用以下命令监听消息:")
                print(f"  mosquitto_sub -h {env_with_cfg.cfg.mqtt.host} -p {env_with_cfg.cfg.mqtt.port} -t '{args.workload_topic}' -v")
            else:
                print(f"[工作负载] ⚠️  警告：无法验证消息发送，但进程仍在运行")
                print(f"[工作负载] 可能的原因:")
                print(f"  1. Broker未正常运行")
                print(f"  2. 工作负载连接Broker失败")
                print(f"  3. 消息发送延迟（等待更长时间后重试）")
                print(f"[工作负载] 提示：可以使用以下命令手动验证:")
                print(f"  mosquitto_sub -h {env_with_cfg.cfg.mqtt.host} -p {env_with_cfg.cfg.mqtt.port} -t '{args.workload_topic}' -C 1")
        else:
            print(f"[工作负载] ⚠️  工作负载可能未正常运行，健康检查将自动恢复")
        
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"\n" + "=" * 80)
        print("错误: 工作负载启动失败，训练无法继续")
        print("=" * 80)
        print(f"错误详情: {e}")
        print("\n请解决以下问题后重新运行:")
        print("1. 确保已安装 emqtt_bench:")
        print("   git clone https://github.com/emqx/emqtt-bench.git")
        print("   cd emqtt-bench && make")
        print("2. 或者设置 EMQTT_BENCH_PATH 环境变量指向 emqtt_bench 可执行文件")
        print("   export EMQTT_BENCH_PATH=/path/to/emqtt_bench")
        print("3. 或者使用 --emqtt-bench-path 参数指定路径")
        print("   --emqtt-bench-path /path/to/emqtt_bench")
        print("\n验证工作负载:")
        print("  python3 script/test_workload.py --duration 10")
        print("=" * 80)
        sys.exit(1)

    # 记录默认配置下的基线性能（训练前）
    record_default_baseline(env, Path(args.save_dir))

    per_beta_anneal_steps = (
        int(args.per_beta_anneal_steps)
        if int(args.per_beta_anneal_steps) > 0
        else int(args.total_timesteps)
    )

    model = make_ddpg_model(
        env=env,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        device=args.device,
        replay_buffer_size=args.replay_buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        utd_ratio=args.utd_ratio,
        policy_delay=args.policy_delay,
        critic_loss=args.critic_loss,
        grad_clip_norm=args.grad_clip_norm,
        target_q_clip=args.target_q_clip,
        use_constraint_weighting=bool(args.use_constraint_weighting),
        action_noise_type=args.action_noise_type,
        action_noise_sigma=args.action_noise_sigma,
        action_noise_theta=args.action_noise_theta,
        use_attention=bool(args.use_attention),
        attention_hidden_dim=args.attention_hidden_dim,
        attention_use_layer_norm=bool(args.attention_use_layer_norm),
        use_per=bool(args.use_per),
        per_alpha=args.per_alpha,
        per_beta0=args.per_beta0,
        per_beta_end=args.per_beta_end,
        per_eps=args.per_eps,
        per_clip_max=args.per_clip_max,
        per_mix_uniform_ratio=args.per_mix_uniform_ratio,
        per_constraint_priority=bool(args.per_constraint_priority),
        per_constraint_scale=args.per_constraint_scale,
        per_beta_anneal_steps=per_beta_anneal_steps,
        use_nstep=bool(args.use_nstep),
        n_step=args.n_step,
        n_step_adaptive=bool(args.n_step_adaptive),
        seed=args.seed,
    )
    if args.use_attention:
        actor_params = sum(p.numel() for p in model.actor.parameters())
        critic_params = sum(p.numel() for p in model.critic.parameters())
        print("[Attention] 已启用 FeatureWiseAttentionExtractor")
        print(f"[Attention] Actor参数量: {actor_params:,}, Critic参数量: {critic_params:,}")
        print(f"[Attention] Actor结构: {model.actor}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置训练日志（保存到 CSV 文件）
    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查 tensorboard 是否可用
    tensorboard_available = False
    if not args.disable_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_available = True
        except ImportError:
            tensorboard_available = False
            print("[警告] tensorboard 未安装，将只使用 stdout 和 csv 日志")
            print("[提示] 安装命令: pip install tensorboard")
    else:
        print("[信息] TensorBoard日志已禁用（节省磁盘空间）")
    
    # 根据可用性和参数配置日志格式，并将logger应用到模型
    # 注意：configure() 返回一个新的logger实例，需要使用 set_logger() 应用到模型
    if tensorboard_available:
        logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    else:
        logger = configure(str(log_dir), ["stdout", "csv"])
    
    # 将配置好的logger应用到模型，这样训练日志才会写入到 progress.csv
    model.set_logger(logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(save_dir),
        name_prefix="ddpg_mosquitto",
        save_replay_buffer=args.save_replay_buffer,  # 根据参数决定是否保存replay buffer
        save_vecnormalize=True,
    )
    
    # 创建checkpoint清理callback（自动删除旧的checkpoint）
    checkpoint_cleanup_callback = CheckpointCleanupCallback(
        save_dir=save_dir,
        max_checkpoints=args.max_checkpoints,
        check_freq=args.save_freq,  # 每次保存checkpoint后检查清理
        verbose=1,
    )
    
    # 创建进度条 callback
    progress_callback = ProgressBarCallback(total_timesteps=args.total_timesteps)
    
    # 创建工作负载健康检查 callback
    # 注意：检查频率设置为每步（check_freq=1），确保Broker重启后立即恢复工作负载
    # Broker重启会导致工作负载断开，需要立即检测并重启
    workload_health_callback = WorkloadHealthCheckCallback(
        workload=workload,
        check_freq=1,  # 每步都检查（确保Broker重启后立即恢复工作负载）
    )
    replay_debug_callback = ReplayDebugCallback(
        check_freq=args.replay_debug_freq,
        verbose=0,
    )
    
    # 创建Mosquitto日志清理callback（可选）
    callbacks = [
        checkpoint_callback,
        checkpoint_cleanup_callback,
        progress_callback,
        workload_health_callback,
        replay_debug_callback,
    ]
    
    if args.cleanup_mosquitto_logs:
        mosquitto_log_cleanup_callback = MosquittoLogCleanupCallback(
            log_dir="/var/log/mosquitto",
            check_freq=args.mosquitto_log_cleanup_freq,
            max_log_files=args.max_mosquitto_log_files,
            verbose=1,
        )
        callbacks.append(mosquitto_log_cleanup_callback)
        print(f"[Mosquitto日志清理] 已启用，每{args.mosquitto_log_cleanup_freq}步清理一次，保留最新{args.max_mosquitto_log_files}个日志文件")

    print(f"\n开始训练 DDPG 模型")
    print(f"总训练步数: {args.total_timesteps:,}")
    print(f"保存目录: {save_dir}")
    print(f"日志目录: {log_dir}")
    print(f"Checkpoint 保存频率: 每 {args.save_freq:,} 步")
    print(f"最多保留checkpoint数: {args.max_checkpoints}")
    print(f"保存replay buffer: {'是' if args.save_replay_buffer else '否（节省磁盘空间）'}")
    print(f"TensorBoard日志: {'启用' if tensorboard_available else '禁用'}")
    print(f"随机种子: {args.seed}")
    print(f"Replay Buffer容量: {args.replay_buffer_size}")
    print(f"Learning starts: {args.learning_starts}")
    print(f"Train freq: {args.train_freq} step")
    print(f"Gradient steps: {args.gradient_steps}")
    print(
        f"Training update: utd_ratio={args.utd_ratio}, "
        f"policy_delay={args.policy_delay}, critic_loss={args.critic_loss}, "
        f"grad_clip_norm={args.grad_clip_norm}, target_q_clip={args.target_q_clip}, "
        f"use_constraint_weighting={bool(args.use_constraint_weighting)}"
    )
    print(f"探索噪声: {args.action_noise_type} (sigma={args.action_noise_sigma}, theta={args.action_noise_theta})")
    print(
        f"Attention: {'启用' if args.use_attention else '关闭'} "
        f"(hidden={args.attention_hidden_dim}, layer_norm={bool(args.attention_use_layer_norm)})"
    )
    print(
        f"PER: {'启用' if args.use_per else '关闭'} "
        f"(alpha={args.per_alpha}, beta_start={args.per_beta0}, beta_end={args.per_beta_end}, "
        f"eps={args.per_eps}, clip_max={args.per_clip_max}, mix_uniform_ratio={args.per_mix_uniform_ratio}, "
        f"constraint_priority={bool(args.per_constraint_priority)}, constraint_scale={args.per_constraint_scale}, "
        f"anneal_steps={per_beta_anneal_steps})"
    )
    print(
        f"N-step: {'启用' if args.use_nstep else '关闭'} "
        f"(n_step={args.n_step}, adaptive={bool(args.n_step_adaptive)})"
    )
    print(
        f"约束模式: {args.constraint_mode}, latency_limit_ms={args.latency_limit_ms}, "
        f"lambda_lr={args.lambda_lr}, penalty_scale={args.penalty_scale}, "
        f"lambda_init={args.constraint_lambda_init}, lambda_max={args.constraint_lambda_max}"
    )
    if args.limit_action_log:
        print(f"Action日志记录间隔: 每{args.action_log_interval}步（节省磁盘空间）")
    print()
    
    # 设置信号处理器，确保 Ctrl+C 时能正确清理资源
    interrupted = {"value": False}
    def signal_handler(signum, frame):
        print("\n\n收到中断信号，正在清理资源...")
        interrupted["value"] = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
        )
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    finally:
        # 确保工作负载被停止
        if workload is not None:
            print("\n停止工作负载...")
            try:
                workload.stop()
                print("工作负载已停止")
            except Exception as e:
                print(f"停止工作负载时出错: {e}")

    # 训练完成后保存最终模型
    final_path = save_dir / "ddpg_mosquitto_final"
    save_model(model, final_path)

    # 关闭环境（ActionThroughputLogger会打印日志统计信息）
    env.close()
    
    # 打印日志文件位置
    action_log_path = save_dir / "action_throughput_log.csv"
    if action_log_path.exists():
        print(f"\n✅ Action和吞吐量日志已保存到: {action_log_path}")
        print(f"   可以使用以下命令查看:")
        print(f"   head -20 {action_log_path}")
        print(f"   或使用Excel/Pandas打开CSV文件进行分析")


if __name__ == "__main__":
    main()
