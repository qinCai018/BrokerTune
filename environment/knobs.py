from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
import os
import subprocess
import time
from pathlib import Path

import numpy as np

"""
和 knobs 相关的数据结构和函数：
- 将 DDPG 连续动作 a_t ∈ [0,1]^n 映射成 Mosquitto 的具体配置项
"""


@dataclass
class BrokerKnobSpace:
    """
    把连续动作向量 a_t ∈ [0,1]^n 映射到具体的 broker 配置空间。

    覆盖的配置包括（与你列出的保持一致）：
    QoS:
      - max_inflight_messages
      - max_inflight_bytes
      - max_queued_messages
      - max_queued_bytes
      - queue_qos0_messages
    内存:
      - memory_limit
      - persistence
      - autosave_interval（代表 autosave_*，0 表示关闭）
    网络延迟:
      - set_tcp_nodelay
    通信和协议层:
      - max_packet_size
      - message_size_limit
    """

    # 下面的范围基于 Mosquitto 官方文档和实际使用场景
    # 范围定义：[最小值, 最大值]
    max_inflight_messages_range: Tuple[int, int] = (0, 2000)
    max_inflight_bytes_range: Tuple[int, int] = (0, 64 * 1024 * 1024)  # 0 或最多 64MB
    max_queued_messages_range: Tuple[int, int] = (0, 20000)
    max_queued_bytes_range: Tuple[int, int] = (0, 128 * 1024 * 1024)  # 0 或最多 128MB

    # 内存上限：0 表示无限制，最大假设 4GB
    memory_limit_range: Tuple[int, int] = (0, 4 * 1024 * 1024 * 1024)
    # autosave 间隔：0 表示关闭，默认值 1800s，最大假设 3600s（1小时）
    autosave_interval_range: Tuple[int, int] = (0, 3600)

    # 报文和消息大小限制：0 或最多 10MB
    # max_packet_size: 当不为0时，最小值应为 20
    max_packet_size_range: Tuple[int, int] = (0, 10 * 1024 * 1024)
    message_size_limit_range: Tuple[int, int] = (0, 10 * 1024 * 1024)

    # 各配置项的量化步长（减少噪声与过大搜索空间带来的抖动）
    max_inflight_messages_step: int = 10
    max_inflight_bytes_step: int = 256 * 1024
    max_queued_messages_step: int = 100
    max_queued_bytes_step: int = 1024 * 1024
    memory_limit_step: int = 64 * 1024 * 1024
    autosave_interval_step: int = 60
    max_packet_size_step: int = 1024
    message_size_limit_step: int = 1024
    
    # 默认值定义（用于环境初始化或作为参考）
    # 这些值对应 Mosquitto 的默认配置
    DEFAULT_MAX_INFLIGHT_MESSAGES: int = 20
    DEFAULT_MAX_INFLIGHT_BYTES: int = 0
    DEFAULT_MAX_QUEUED_MESSAGES: int = 1000
    DEFAULT_MAX_QUEUED_BYTES: int = 0
    DEFAULT_QUEUE_QOS0_MESSAGES: bool = False
    DEFAULT_MEMORY_LIMIT: int = 0  # 0 表示无限制
    DEFAULT_PERSISTENCE: bool = False
    DEFAULT_AUTOSAVE_INTERVAL: int = 1800
    DEFAULT_SET_TCP_NODELAY: bool = False
    DEFAULT_MAX_PACKET_SIZE: int = 0
    DEFAULT_MESSAGE_SIZE_LIMIT: int = 0

    @property
    def action_dim(self) -> int:
        """
        动作向量维度：
          0: max_inflight_messages
          1: max_inflight_bytes
          2: max_queued_messages
          3: max_queued_bytes
          4: queue_qos0_messages (bool)
          5: memory_limit
          6: persistence (bool)
          7: autosave_interval
          8: set_tcp_nodelay (bool)
          9: max_packet_size
         10: message_size_limit
        """
        return 11

    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """
        将归一化到 [0,1] 的动作向量映射为具体配置字典。

        注意：
        - 对于允许 0 表示“无限制/关闭”的项，当动作非常接近 0 时会直接映射为 0
        - 布尔项通过 0.5 阈值进行取整
        """
        # 确保动作是numpy数组且形状正确
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # 验证动作维度
        if action.shape[0] != self.action_dim:
            raise ValueError(f"动作维度不匹配: 期望 {self.action_dim}, 得到 {action.shape[0]}")
        
        # Clip到有效范围并验证
        a = np.clip(action, 0.0, 1.0).astype(np.float32)
        
        # 检查是否有NaN/Inf
        if np.any(np.isnan(a)) or np.any(np.isinf(a)):
            print(f"[BrokerKnobSpace] 警告: 检测到无效动作值（NaN/Inf），使用0.5作为默认值")
            a = np.nan_to_num(a, nan=0.5, posinf=1.0, neginf=0.0)

        def _interp_with_zero(v: float, low: int, high: int, zero_eps: float = 0.01) -> int:
            """
            对于 0 表示无限制/关闭的配置：
            - v < zero_eps/2 时直接返回 0（确保编码时zero_eps/2映射为0）
            - 否则在线性插值 [low, high]（low 通常也是 0）
            
            注意：
            - 编码时，0值映射为zero_eps/2（0.005）
            - 解码时，v < zero_eps/2（0.005）→ 0
            - 解码时，v >= zero_eps/2（0.005）→ 正常插值，使用round避免浮点数精度问题
            - zero_eps设置为0.01，这样20/2000=0.01不会被误判为0
            """
            if v < zero_eps / 2.0:
                return 0
            # 使用round避免浮点数精度问题（例如0.01 * 2000 = 19.9999996）
            return int(round(low + v * (high - low)))

        def _interp(v: float, low: int, high: int) -> int:
            return int(low + v * (high - low))

        def _quantize(value: int, step: int, low: int, high: int) -> int:
            if value == 0:
                return 0
            if step <= 1:
                return int(min(max(value, low), high))
            quantized = int(round(value / step) * step)
            if quantized == 0:
                quantized = step
            return int(min(max(quantized, low if low > 0 else step), high))

        # QoS 相关
        max_inflight_messages = _interp_with_zero(
            a[0], *self.max_inflight_messages_range
        )
        max_inflight_messages = _quantize(
            max_inflight_messages,
            self.max_inflight_messages_step,
            *self.max_inflight_messages_range
        )
        max_inflight_bytes = _interp_with_zero(
            a[1], *self.max_inflight_bytes_range
        )
        max_inflight_bytes = _quantize(
            max_inflight_bytes,
            self.max_inflight_bytes_step,
            *self.max_inflight_bytes_range
        )
        max_queued_messages = _interp_with_zero(
            a[2], *self.max_queued_messages_range
        )
        max_queued_messages = _quantize(
            max_queued_messages,
            self.max_queued_messages_step,
            *self.max_queued_messages_range
        )
        max_queued_bytes = _interp_with_zero(
            a[3], *self.max_queued_bytes_range
        )
        max_queued_bytes = _quantize(
            max_queued_bytes,
            self.max_queued_bytes_step,
            *self.max_queued_bytes_range
        )
        queue_qos0_messages = bool(a[4] >= 0.5)

        # 内存 / 持久化
        memory_limit = _interp_with_zero(
            a[5], *self.memory_limit_range
        )
        memory_limit = _quantize(
            memory_limit,
            self.memory_limit_step,
            *self.memory_limit_range
        )
        persistence = bool(a[6] >= 0.5)
        autosave_interval = _interp_with_zero(
            a[7], *self.autosave_interval_range
        )
        autosave_interval = _quantize(
            autosave_interval,
            self.autosave_interval_step,
            *self.autosave_interval_range
        )

        # 网络 / 协议层
        set_tcp_nodelay = bool(a[8] >= 0.5)
        # max_packet_size: 当不为0时，最小值应为 20
        max_packet_size_raw = _interp_with_zero(
            a[9], *self.max_packet_size_range
        )
        max_packet_size_raw = _quantize(
            max_packet_size_raw,
            self.max_packet_size_step,
            *self.max_packet_size_range
        )
        if max_packet_size_raw > 0 and max_packet_size_raw < 20:
            max_packet_size = 20
        else:
            max_packet_size = max_packet_size_raw
        message_size_limit = _interp_with_zero(
            a[10], *self.message_size_limit_range
        )
        message_size_limit = _quantize(
            message_size_limit,
            self.message_size_limit_step,
            *self.message_size_limit_range
        )

        return {
            # QoS
            "max_inflight_messages": max_inflight_messages,
            "max_inflight_bytes": max_inflight_bytes,
            "max_queued_messages": max_queued_messages,
            "max_queued_bytes": max_queued_bytes,
            "queue_qos0_messages": queue_qos0_messages,
            # 内存
            "memory_limit": memory_limit,
            "persistence": persistence,
            "autosave_interval": autosave_interval,
            # 网络延迟
            "set_tcp_nodelay": set_tcp_nodelay,
            # 协议/报文
            "max_packet_size": max_packet_size,
            "message_size_limit": message_size_limit,
        }
    
    def get_default_knobs(self) -> Dict[str, Any]:
        """
        返回 Mosquitto 的默认配置字典。
        
        这些默认值可用于：
        - 环境初始化时的初始配置
        - 作为强化学习训练的基准配置
        - 重置环境到默认状态
        
        Returns:
            包含所有参数默认值的字典
        """
        return {
            # QoS
            "max_inflight_messages": self.DEFAULT_MAX_INFLIGHT_MESSAGES,
            "max_inflight_bytes": self.DEFAULT_MAX_INFLIGHT_BYTES,
            "max_queued_messages": self.DEFAULT_MAX_QUEUED_MESSAGES,
            "max_queued_bytes": self.DEFAULT_MAX_QUEUED_BYTES,
            "queue_qos0_messages": self.DEFAULT_QUEUE_QOS0_MESSAGES,
            # 内存
            "memory_limit": self.DEFAULT_MEMORY_LIMIT,
            "persistence": self.DEFAULT_PERSISTENCE,
            "autosave_interval": self.DEFAULT_AUTOSAVE_INTERVAL,
            # 网络延迟
            "set_tcp_nodelay": self.DEFAULT_SET_TCP_NODELAY,
            # 协议/报文
            "max_packet_size": self.DEFAULT_MAX_PACKET_SIZE,
            "message_size_limit": self.DEFAULT_MESSAGE_SIZE_LIMIT,
        }
    
    def encode_knobs(self, knobs: Dict[str, Any]) -> np.ndarray:
        """
        将配置字典编码为归一化的action向量 [0, 1]。
        
        这是 decode_action 的逆操作，用于将默认配置编码为action向量。
        
        Args:
            knobs: 配置字典
            
        Returns:
            归一化的action向量，形状为 (action_dim,)
        """
        def _encode_with_zero(value: int, low: int, high: int, zero_eps: float = 0.01) -> float:
            """
            对于允许0表示无限制的配置，将配置值编码为action。
            
            编码策略：
            - value = 0 → action = zero_eps / 2.0（确保解码时能被识别为0）
            - value > 0 → action = (value - low) / (high - low)
            
            注意：zero_eps设置为0.01，这样20/2000=0.01不会被误判为0
            """
            if value == 0:
                return zero_eps / 2.0  # 映射到小于zero_eps的值（0.005）
            if high == low:
                return 0.5  # 避免除零
            return (value - low) / (high - low)
        
        def _encode(value: int, low: int, high: int) -> float:
            """普通线性编码"""
            if high == low:
                return 0.5  # 避免除零
            return (value - low) / (high - low)
        
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        # QoS 相关
        action[0] = _encode_with_zero(
            knobs.get("max_inflight_messages", self.DEFAULT_MAX_INFLIGHT_MESSAGES),
            *self.max_inflight_messages_range
        )
        action[1] = _encode_with_zero(
            knobs.get("max_inflight_bytes", self.DEFAULT_MAX_INFLIGHT_BYTES),
            *self.max_inflight_bytes_range
        )
        action[2] = _encode_with_zero(
            knobs.get("max_queued_messages", self.DEFAULT_MAX_QUEUED_MESSAGES),
            *self.max_queued_messages_range
        )
        action[3] = _encode_with_zero(
            knobs.get("max_queued_bytes", self.DEFAULT_MAX_QUEUED_BYTES),
            *self.max_queued_bytes_range
        )
        action[4] = 1.0 if knobs.get("queue_qos0_messages", self.DEFAULT_QUEUE_QOS0_MESSAGES) else 0.0
        
        # 内存相关
        action[5] = _encode_with_zero(
            knobs.get("memory_limit", self.DEFAULT_MEMORY_LIMIT),
            *self.memory_limit_range
        )
        action[6] = 1.0 if knobs.get("persistence", self.DEFAULT_PERSISTENCE) else 0.0
        action[7] = _encode_with_zero(
            knobs.get("autosave_interval", self.DEFAULT_AUTOSAVE_INTERVAL),
            *self.autosave_interval_range
        )
        
        # 网络延迟
        action[8] = 1.0 if knobs.get("set_tcp_nodelay", self.DEFAULT_SET_TCP_NODELAY) else 0.0
        
        # 通信和协议层
        action[9] = _encode_with_zero(
            knobs.get("max_packet_size", self.DEFAULT_MAX_PACKET_SIZE),
            *self.max_packet_size_range
        )
        action[10] = _encode_with_zero(
            knobs.get("message_size_limit", self.DEFAULT_MESSAGE_SIZE_LIMIT),
            *self.message_size_limit_range
        )
        
        # 确保在有效范围内
        action = np.clip(action, 0.0, 1.0)
        
        return action
    
    def get_default_action(self) -> np.ndarray:
        """
        返回默认配置对应的action向量。
        
        Returns:
            默认配置的归一化action向量
        """
        return self.encode_knobs(self.get_default_knobs())


def apply_knobs(knobs: Dict[str, Any], dry_run: bool = None, force_restart: bool = None) -> bool:
    """
    将解码后的 broker 配置真正作用到 Mosquitto。

    新行为（使用独立配置文件）：
      1. 创建一个完整的独立配置文件，包含所有必需的配置项
         - 配置文件路径：MOSQUITTO_TUNER_CONFIG（默认：./broker_tuner.conf）
         - 包含基础配置（listener、allow_anonymous、sys_interval等）
         - 包含训练过程中调整的配置（knobs中的配置）
      2. 使用 mosquitto -c xxx.conf 方式启动，而不是 systemctl
         - 先停止现有的 mosquitto 进程（systemctl stop 或 pkill）
         - 使用新配置文件启动 mosquitto（后台运行）
         - 每次配置变化都会完全重启 mosquitto

    参数:
      dry_run: 如果为 True，只打印配置信息，不实际写入文件或重启服务。
               如果为 None，则从环境变量 BROKER_TUNER_DRY_RUN 读取（默认为 False）
      force_restart: 已废弃，现在总是完全重启（保留此参数以兼容旧代码）
    
    Returns:
      bool: 总是返回 True（表示完全重启）
    """
    # 检查是否启用测试模式
    if dry_run is None:
        dry_run = os.environ.get("BROKER_TUNER_DRY_RUN", "false").lower() in ("true", "1", "yes")
    
    # 配置文件路径（使用environment/config目录下的独立配置文件）
    # 默认路径：~/userDir/BrokerTuner/environment/config/broker_tuner.conf
    # 如果环境变量设置了MOSQUITTO_TUNER_CONFIG，则使用指定的路径
    config_path_str = os.environ.get("MOSQUITTO_TUNER_CONFIG")
    if config_path_str is None:
        # 使用environment/config目录下的配置文件
        # Path(__file__) 指向 ~/userDir/BrokerTuner/environment/knobs.py
        env_dir = Path(__file__).parent  # environment目录
        config_dir = env_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        config_path_str = str(config_dir / "broker_tuner.conf")
    config_path = Path(config_path_str).resolve()  # 使用绝对路径

    # 从模板文件开始构建完整的配置文件内容
    # 这是一个独立的完整配置文件，包含所有必需的配置项
    env_dir = Path(__file__).parent  # environment目录
    template_path = env_dir / "config" / "broker_template.conf"

    # 读取模板文件
    try:
        template_content = template_path.read_text(encoding="utf-8")
        lines = template_content.splitlines()
    except FileNotFoundError:
        # 如果模板文件不存在，使用基础配置
        print(f"警告: 模板文件 {template_path} 不存在，使用基础配置")
        lines = [
            "# BrokerTuner - 完整的 Mosquitto Broker 配置文件",
            "# 此文件可以直接用于启动 Broker: mosquitto -c broker_tuner.conf",
            "# 自动生成，训练过程中会保留最佳效果的配置",
            "# 请不要手工修改，该文件可能会被覆盖",
            "",
            "# PID 文件位置（用于跟踪进程）",
            f"pid_file {str(config_path.parent / 'mosquitto_broker_tuner.pid')}",
            "",
            "# 监听端口 1883（MQTT 标准端口）",
            "listener 1883",
            "",
            "# 允许匿名连接（用于测试和开发）",
            "allow_anonymous true",
            "",
            "# 持久化配置",
            "persistence false",
            "",
            "# 日志配置",
            "log_type none",
            "",
            "# $SYS 主题配置",
            "sys_interval 1",
        ]

    def add_line(key: str, value: Any) -> None:
        """添加配置行"""
        if isinstance(value, bool):
            v_str = "true" if value else "false"
        else:
            v_str = str(int(value))
        lines.append(f"{key} {v_str}")

    # 在模板基础上添加训练过程中动态调整的配置参数
    # 模板文件已经包含了基础配置，这里只添加可调参数

    # 首先处理持久化配置（如果在knobs中指定）
    if "persistence" in knobs:
        # 查找并替换模板中的persistence配置
        persistence_value = "true" if knobs['persistence'] else "false"
        for i, line in enumerate(lines):
            if line.strip().startswith("persistence") and not line.strip().startswith("#"):
                lines[i] = f"persistence {persistence_value}"
                break
        else:
            # 如果模板中没有，添加一行
            lines.append(f"persistence {persistence_value}")

    lines.append("")
    lines.append("# ============================================")
    lines.append("# 训练过程中动态调整的配置参数")
    lines.append("# ============================================")
    lines.append("")

    # QoS 相关配置
    if "max_inflight_messages" in knobs:
        add_line("max_inflight_messages", knobs["max_inflight_messages"])
    if "max_inflight_bytes" in knobs and knobs["max_inflight_bytes"] > 0:
        add_line("max_inflight_bytes", knobs["max_inflight_bytes"])
    if "max_queued_messages" in knobs:
        add_line("max_queued_messages", knobs["max_queued_messages"])
    if "max_queued_bytes" in knobs and knobs["max_queued_bytes"] > 0:
        add_line("max_queued_bytes", knobs["max_queued_bytes"])
    if "queue_qos0_messages" in knobs:
        add_line("queue_qos0_messages", knobs["queue_qos0_messages"])

    # 内存配置
    if "memory_limit" in knobs and knobs["memory_limit"] > 0:
        add_line("memory_limit", knobs["memory_limit"])
    if "autosave_interval" in knobs and knobs["autosave_interval"] > 0:
        add_line("autosave_interval", knobs["autosave_interval"])

    # 网络 / 协议配置
    if "set_tcp_nodelay" in knobs:
        add_line("set_tcp_nodelay", knobs["set_tcp_nodelay"])
    if "max_packet_size" in knobs and knobs["max_packet_size"] > 0:
        # max_packet_size 最小值应为 20
        max_packet_size = max(20, knobs["max_packet_size"])
        add_line("max_packet_size", max_packet_size)
    if "message_size_limit" in knobs and knobs["message_size_limit"] > 0:
        add_line("message_size_limit", knobs["message_size_limit"])

    # 测试模式：只打印配置信息，不实际写入文件
    if dry_run:
        print(f"[DRY RUN] 将应用的配置 knobs: {knobs}")
        print(f"[DRY RUN] 配置文件内容（不会实际写入）:")
        print("\n".join(lines))
        return True  # 测试模式返回 True（视为完全重启）

    # 写入配置文件
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_content = "\n".join(lines) + "\n"
        config_path.write_text(config_content, encoding="utf-8")
        print(f"[apply_knobs] ✅ 配置文件已写入: {config_path}")
    except OSError as exc:
        raise RuntimeError(f"写入 Mosquitto 配置文件失败: {config_path} ({exc})") from exc

    # 停止现有的 mosquitto 进程
    def _stop_mosquitto() -> None:
        """停止现有的 mosquitto 进程"""
        print("[apply_knobs] 停止现有的 mosquitto 进程...")
        
        # 方法1: 尝试使用 systemctl stop（如果服务正在运行）
        try:
            result = subprocess.run(
                ["systemctl", "stop", "mosquitto"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print("[apply_knobs] ✅ 已通过 systemctl stop 停止 mosquitto 服务")
                time.sleep(1)  # 等待服务停止
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            pass  # systemctl 可能不可用或服务未运行
        
        # 方法2: 使用 pkill 停止所有 mosquitto 进程（包括使用 -c 启动的）
        try:
            # 先尝试优雅停止（SIGTERM）
            subprocess.run(
                ["pkill", "-TERM", "-f", "mosquitto"],
                capture_output=True,
                timeout=5
            )
            time.sleep(2)  # 等待进程退出
            
            # 检查是否还有进程在运行
            result = subprocess.run(
                ["pgrep", "-f", "mosquitto"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                # 如果还有进程，强制终止（SIGKILL）
                print("[apply_knobs] 仍有进程运行，强制终止...")
                subprocess.run(
                    ["pkill", "-KILL", "-f", "mosquitto"],
                    capture_output=True,
                    timeout=5
                )
                time.sleep(1)
            
            print("[apply_knobs] ✅ 已停止所有 mosquitto 进程")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # 等待端口1883释放
        print("[apply_knobs] 等待端口1883释放...")
        for i in range(10):
            try:
                result = subprocess.run(
                    ["netstat", "-tln"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if ":1883" not in result.stdout:
                    break  # 端口已释放
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # 如果 netstat 不可用，尝试使用 ss
                try:
                    result = subprocess.run(
                        ["ss", "-tln"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if ":1883" not in result.stdout:
                        break  # 端口已释放
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
            time.sleep(0.5)
    
    # 启动 mosquitto（使用独立配置文件）
    def _start_mosquitto() -> None:
        """使用独立配置文件启动 mosquitto"""
        print(f"[apply_knobs] 使用配置文件启动 mosquitto: {config_path}")
        
        # 检查 mosquitto 可执行文件是否存在
        mosquitto_cmd = None
        for path in ["/usr/sbin/mosquitto", "/usr/bin/mosquitto", "mosquitto"]:
            try:
                result = subprocess.run(
                    ["which", path] if path == "mosquitto" else ["test", "-f", path],
                    capture_output=True,
                    timeout=2
                )
                if result.returncode == 0 or path == "mosquitto":
                    mosquitto_cmd = path
                    break
            except:
                continue
        
        if mosquitto_cmd is None:
            # 尝试直接使用 mosquitto（可能在 PATH 中）
            mosquitto_cmd = "mosquitto"
        
        # 启动 mosquitto（后台运行）
        # -c: 指定配置文件
        # -d: 后台运行（daemon模式）
        try:
            result = subprocess.run(
                [mosquitto_cmd, "-c", str(config_path), "-d"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip() or "未知错误"
                raise RuntimeError(
                    f"启动 mosquitto 失败 (退出码: {result.returncode})\n"
                    f"错误信息: {error_msg}\n"
                    f"配置文件: {config_path}\n"
                    f"请检查配置文件是否有语法错误。\n"
                    f"提示: 可以手动测试: {mosquitto_cmd} -c {config_path}"
                )
            
            print(f"[apply_knobs] ✅ mosquitto 已启动（后台运行）")
            
            # 等待一下，让进程启动
            time.sleep(2)
            
            # 验证进程是否运行
            result = subprocess.run(
                ["pgrep", "-f", f"mosquitto.*{config_path.name}"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                pid = result.stdout.strip().split()[0]
                print(f"[apply_knobs] ✅ mosquitto 进程已运行（PID: {pid}）")
                # 更新环境变量中的PID
                os.environ["MOSQUITTO_PID"] = pid
            else:
                print("[apply_knobs] ⚠️  警告: 无法验证 mosquitto 进程是否运行")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"启动 mosquitto 超时: {mosquitto_cmd} -c {config_path}")
        except FileNotFoundError:
            raise RuntimeError(
                f"找不到 mosquitto 可执行文件\n"
                f"请确保已安装 mosquitto: sudo apt install mosquitto"
            )
    
    # 执行停止和启动
    try:
        _stop_mosquitto()
        _start_mosquitto()
        return True  # 总是返回 True（表示完全重启）
    except Exception as exc:
        raise RuntimeError(
            f"应用配置失败: {exc}\n"
            f"配置文件: {config_path}\n"
            f"提示: 可以手动检查配置文件: cat {config_path}"
        ) from exc
