"""
模型训练入口：
- 创建 MosquittoBrokerEnv 环境
- 使用自定义 Policy 的 DDPG 进行训练
- 定期保存模型

使用示例：
    python -m tuner.train --total-timesteps 100000 --save-dir ./checkpoints
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

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
        default=0.00001,
        help="目标网络软更新系数，默认：0.00001",
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=0.00001,
        help="Actor学习率，默认：0.00001",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=0.00001,
        help="Critic学习率，默认：0.00001",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="折扣因子，默认：0.9",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="训练批次大小，默认：16",
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
    
    特殊处理：第一步使用默认配置的action
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
        
        # 标记是否是第一步（每个episode的第一步使用默认action）
        self._is_first_step = True
        
        # 获取默认action（对应Mosquitto默认配置）
        self._default_action = None
        # 尝试获取knob_space（可能需要unwrapped）
        env_for_knob_space = env
        for _ in range(5):  # 最多尝试5层
            if hasattr(env_for_knob_space, 'knob_space'):
                self._default_action = env_for_knob_space.knob_space.get_default_action()
                self._cached_knob_space = env_for_knob_space.knob_space  # 缓存knob_space
                print(f"[ActionThroughputLogger] 已获取默认action（对应Mosquitto默认配置）")
                break
            elif hasattr(env_for_knob_space, 'unwrapped'):
                env_for_knob_space = env_for_knob_space.unwrapped
            elif hasattr(env_for_knob_space, 'env'):
                env_for_knob_space = env_for_knob_space.env
            else:
                break
        
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
                    ["throughput", "reward"]
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
        self._is_first_step = True  # 标记为第一步，将使用默认action
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """执行一步，记录action和吞吐量"""
        self.current_step += 1
        
        # 第一步使用默认action（对应Mosquitto默认配置）
        if self._is_first_step and self._default_action is not None:
            print(f"[ActionThroughputLogger] 第一步使用默认配置action（episode {self.current_episode}）")
            action = self._default_action.copy()
            self._is_first_step = False
        
        # 执行环境step
        if self.current_step <= 3 or self.current_step % 20 == 0:
            print(f"[ActionThroughputLogger] 执行env.step()（步数: {self.current_step}）...")
        result = self.env.step(action)
        
        # 根据log_interval决定是否记录日志
        should_log = (self.current_step % self.log_interval == 0) or (self.current_step <= 3)
        
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
            if not hasattr(self, '_cached_knob_space'):
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
        
        # 记录到CSV文件（根据log_interval决定是否记录）
        if should_log:
            if self.current_step <= 3 or self.current_step % 20 == 0:
                print(f"[ActionThroughputLogger] 开始写入CSV文件...")
            try:
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    # 将action转换为列表（如果是numpy数组）
                    action_list = action.tolist() if hasattr(action, 'tolist') else list(action)
                # 行数据：步数、episode、11个action值（归一化）、11个解码后的配置值、吞吐量、奖励
                # 注意：扩展状态向量后，可以添加更多指标到CSV
                row = (
                    [self.current_step, self.current_episode] +
                    action_list +
                    decoded_values +
                    [throughput, reward]
                )
                writer.writerow(row)
                f.flush()  # 确保立即写入磁盘
                import os
                os.fsync(f.fileno())  # 强制同步到磁盘
                if self.current_step <= 3 or self.current_step % 20 == 0:
                    print(f"[ActionThroughputLogger] CSV写入完成")
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


def main() -> None:
    args = parse_args()

    env_cfg = EnvConfig()
    
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

    model = make_ddpg_model(
        env=env,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        device=args.device,
    )

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
    
    # 创建Mosquitto日志清理callback（可选）
    callbacks = [
        checkpoint_callback,
        checkpoint_cleanup_callback,
        progress_callback,
        workload_health_callback,
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

