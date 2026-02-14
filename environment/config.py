import os
from dataclasses import dataclass, field
from typing import List
"""
环境与采样配置（MQTT、/proc、状态/动作维度等）
"""

@dataclass
class MQTTConfig:
    """Mosquitto 相关的 MQTT 采样配置"""

    host: str = "127.0.0.1"
    port: int = 1883
    client_id: str = "broker_tuner_monitor"
    # 订阅 broker 运行指标
    topics: List[str] = ("$SYS/#",)
    keepalive: int = 30
    timeout_sec: float = 12.0  # 等待一轮采样的超时时间（默认12秒；若sys_interval更小可调低以加速）
    rate_min_interval_sec: float = 5.0  # 速率估算的最小时间间隔
    rate_min_samples: int = 2  # 速率估算的最小样本数
    rate_1min_divisor: float = 60.0  # 若$SYS/broker/load/messages/received/1min为每分钟消息数则用60转换为每秒
    rate_1min_window_sec: float = 60.0  # 1min指标通常需要至少60秒uptime才稳定
    sample_wait_for_topics: List[str] = field(
        default_factory=lambda: [
            "$SYS/broker/messages/received",
            "$SYS/broker/clients/connected",
            "$SYS/broker/uptime",
        ]
    )
    sample_wait_for_derived_rate: bool = True
    sample_poll_interval_sec: float = 0.1


@dataclass
class ProcConfig:
    """通过 /proc 采样进程 CPU / 内存 / 上下文切换"""

    pid: int = 0  # 将在 __post_init__ 中自动检测
    # CPU 归一化的参考：如逻辑核数*100
    cpu_norm: float = 400.0
    mem_norm: float = 1024 * 1024 * 1024  # 1 GiB
    ctxt_norm: float = 1e6
    
    def __post_init__(self):
        """自动检测 Mosquitto PID（如果未设置）"""
        if self.pid == 0:
            # 优先使用环境变量
            env_pid = os.environ.get("MOSQUITTO_PID")
            if env_pid:
                try:
                    self.pid = int(env_pid)
                    return
                except ValueError:
                    pass
            
            # 如果环境变量未设置，尝试自动检测
            import subprocess
            try:
                # 使用 pgrep 获取第一个 mosquitto 进程 PID
                result = subprocess.run(
                    ["pgrep", "-o", "mosquitto"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0 and result.stdout.strip():
                    detected_pid = int(result.stdout.strip())
                    self.pid = detected_pid
                    # 设置环境变量，以便后续使用
                    os.environ["MOSQUITTO_PID"] = str(detected_pid)
                    return
            except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
                pass
            
            # 如果都失败了，设置为 0（会在 read_proc_metrics 中报错）
            self.pid = 0


@dataclass
class EnvConfig:
    """
    强化学习环境的整体配置：
    - 状态采样
    - 步长 / 回合长度
    """

    mqtt: MQTTConfig = field(default_factory=MQTTConfig)
    proc: ProcConfig = field(default_factory=ProcConfig)

    # 每一步 action 之后等待系统稳定的时间（秒）
    # 建议 >= sys_interval(10s)，以避免重复采样旧指标
    step_interval_sec: float = 12.0
    
    # Broker 重启后的稳定等待时间（秒）
    # 当 Broker 完全重启（restart）时，需要等待更长时间让系统稳定
    # 实际等待时间会根据 Broker 是否就绪动态调整（最多等待此时间）
    # 注意：会验证端口1883是否监听，确保Broker真正就绪
    broker_restart_stable_sec: float = 20.0  # 增加到20秒，确保Broker完全启动并监听端口
    
    # Broker 重载后的稳定等待时间（秒）
    # 当 Broker 只是重载配置（reload）时，等待时间可以较短
    broker_reload_stable_sec: float = 3.0

    # 是否在每个 episode 重置时应用默认配置
    apply_default_on_reset: bool = True

    # 是否在每个 episode 重置时更新奖励基线
    baseline_per_episode: bool = True
    baseline_min_throughput: float = 0.05  # msg_rate_norm 的最小可接受值，避免基线被采到接近0导致奖励失真
    baseline_min_clients_norm: float = 0.001  # clients_norm 的最小可接受值（1/1000）
    baseline_max_attempts: int = 5
    baseline_retry_sleep_sec: float = 2.0

    # 每个 episode 的最大步数
    max_steps: int = 200

    # 状态维度：可以根据实际提取的指标调整
    # 这里假设：
    # [0] broker 当前连接数
    # [1] broker 消息速率（msg/s）
    # [2] CPU 使用占比
    # [3] RSS 内存占比
    # [4] 每秒上下文切换数占比
    # [5] P50 端到端延迟（ms）
    # [6] P95 端到端延迟（ms）
    # [7] 队列深度（归一化）
    # [8] 最近5步平均吞吐量（滑动窗口）
    # [9] 最近5步平均延迟（滑动窗口）
    state_dim: int = 10

    # 动作向量维度：由 knobs.py 中定义的可调节参数个数决定
    # 当前为 11，对应 BrokerKnobSpace.action_dim
    action_dim: int = 11

    # 奖励计算参数
    reward_scale: float = 5.0
    reward_weight_base: float = 0.8
    reward_weight_step: float = 0.2
    reward_weight_latency_base: float = 0.2
    reward_weight_latency_step: float = 0.1
    reward_clip: float = 5.0
    reward_delta_clip: float = 2.0
    reward_use_tanh: bool = True
    reward_latency_floor_norm: float = 0.01
    failed_step_penalty: float = -3.0
    max_consecutive_failures: int = 3
    latency_fallback_p50_ms: float = 20.0
    latency_fallback_p95_ms: float = 80.0
    enable_latency_probe: bool = True
