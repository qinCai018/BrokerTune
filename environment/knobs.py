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

        # QoS 相关
        max_inflight_messages = _interp_with_zero(
            a[0], *self.max_inflight_messages_range
        )
        max_inflight_bytes = _interp_with_zero(
            a[1], *self.max_inflight_bytes_range
        )
        max_queued_messages = _interp_with_zero(
            a[2], *self.max_queued_messages_range
        )
        max_queued_bytes = _interp_with_zero(
            a[3], *self.max_queued_bytes_range
        )
        queue_qos0_messages = bool(a[4] >= 0.5)

        # 内存 / 持久化
        memory_limit = _interp_with_zero(
            a[5], *self.memory_limit_range
        )
        persistence = bool(a[6] >= 0.5)
        autosave_interval = _interp_with_zero(
            a[7], *self.autosave_interval_range
        )

        # 网络 / 协议层
        set_tcp_nodelay = bool(a[8] >= 0.5)
        # max_packet_size: 当不为0时，最小值应为 20
        max_packet_size_raw = _interp_with_zero(
            a[9], *self.max_packet_size_range
        )
        if max_packet_size_raw > 0 and max_packet_size_raw < 20:
            max_packet_size = 20
        else:
            max_packet_size = max_packet_size_raw
        message_size_limit = _interp_with_zero(
            a[10], *self.message_size_limit_range
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

    默认行为（可通过环境变量调整）：
      1. 把 knobs 写入一个单独的配置文件：MOSQUITTO_TUNER_CONFIG
         - 默认：/etc/mosquitto/conf.d/broker_tuner.conf
      2. 调用 systemctl 重载/重启 mosquitto 服务：
         - 如果 force_restart=True 或环境变量 BROKER_TUNER_FORCE_RESTART=true：
           直接使用 systemctl restart mosquitto（完全重启）
         - 否则优先使用：systemctl reload mosquitto（重载配置）
         - 如果 reload 失败，回退到：systemctl restart mosquitto

    参数:
      dry_run: 如果为 True，只打印配置信息，不实际写入文件或重启服务。
               如果为 None，则从环境变量 BROKER_TUNER_DRY_RUN 读取（默认为 False）
      force_restart: 如果为 True，强制使用 restart 而不是 reload。
                     如果为 None，则从环境变量 BROKER_TUNER_FORCE_RESTART 读取（默认为 False）
                     某些配置（如 persistence、memory_limit）可能需要完全重启才能生效

    注意：
      - 这些操作通常需要 root 权限，你需要确保运行该进程的用户有相应权限
      - 如果你已有自己的管理脚本，也可以忽略这里的实现，改成调用自定义脚本
      - 在测试模式下（dry_run=True），不会实际修改系统配置，适合观察交互结果
      - restart 会断开所有现有连接，reload 不会断开连接但某些配置可能不生效
    
    Returns:
      bool: 如果使用了 restart（完全重启）返回 True，如果使用了 reload（重载）返回 False
    """
    # 检查是否启用测试模式
    if dry_run is None:
        dry_run = os.environ.get("BROKER_TUNER_DRY_RUN", "false").lower() in ("true", "1", "yes")
    
    # 检查是否强制使用 restart
    if force_restart is None:
        force_restart = os.environ.get("BROKER_TUNER_FORCE_RESTART", "false").lower() in ("true", "1", "yes")
    
    # 检查是否需要强制重启（某些配置需要完全重启才能生效）
    if not force_restart:
        # 这些配置项需要完全重启才能生效
        restart_required_configs = ["persistence", "memory_limit", "autosave_interval"]
        if any(key in knobs for key in restart_required_configs):
            force_restart = True
    
    config_path_str = os.environ.get(
        "MOSQUITTO_TUNER_CONFIG", "/etc/mosquitto/conf.d/broker_tuner.conf"
    )
    config_path = Path(config_path_str)

    lines = [
        "# 自动生成：BrokerTuner knobs",
        "# 请不要手工修改，该文件可能会被覆盖",
        "",
    ]

    def add_line(key: str, value: Any) -> None:
        if isinstance(value, bool):
            v_str = "true" if value else "false"
        else:
            v_str = str(int(value))
        lines.append(f"{key} {v_str}")

    # QoS 相关
    if "max_inflight_messages" in knobs:
        add_line("max_inflight_messages", knobs["max_inflight_messages"])
    if "max_inflight_bytes" in knobs:
        add_line("max_inflight_bytes", knobs["max_inflight_bytes"])
    if "max_queued_messages" in knobs:
        add_line("max_queued_messages", knobs["max_queued_messages"])
    if "max_queued_bytes" in knobs:
        add_line("max_queued_bytes", knobs["max_queued_bytes"])
    if "queue_qos0_messages" in knobs:
        add_line("queue_qos0_messages", knobs["queue_qos0_messages"])

    # 内存 / 持久化
    if "memory_limit" in knobs:
        add_line("memory_limit", knobs["memory_limit"])
    if "persistence" in knobs:
        add_line("persistence", knobs["persistence"])
    if "autosave_interval" in knobs:
        # autosave_interval 仅在 persistence=true 时有意义，但这里统一写入
        add_line("autosave_interval", knobs["autosave_interval"])

    # 网络 / 协议
    if "set_tcp_nodelay" in knobs:
        add_line("set_tcp_nodelay", knobs["set_tcp_nodelay"])
    if "max_packet_size" in knobs:
        max_packet_size = knobs["max_packet_size"]
        # max_packet_size 为 0 表示无限制，但 Mosquitto 可能不接受 0
        # 如果为 0，不写入该配置项（使用默认值）
        if max_packet_size > 0:
            add_line("max_packet_size", max_packet_size)
    if "message_size_limit" in knobs:
        message_size_limit = knobs["message_size_limit"]
        # message_size_limit 为 0 表示无限制，但 Mosquitto 可能不接受 0
        # 如果为 0，不写入该配置项（使用默认值）
        if message_size_limit > 0:
            add_line("message_size_limit", message_size_limit)

    # 测试模式：只打印配置信息，不实际写入文件
    if dry_run:
        print(f"[DRY RUN] 将应用的配置 knobs: {knobs}")
        print(f"[DRY RUN] 配置文件内容（不会实际写入）:")
        print("\n".join(lines))
        return False  # 测试模式返回 False（视为 reload）

    # 写入配置文件
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_content = "\n".join(lines) + "\n"
        config_path.write_text(config_content, encoding="utf-8")
        
        # 注意：不进行语法检查，因为 Mosquitto 1.6.9 不支持 -t 选项
        # systemctl reload/restart 会自动验证配置，如果配置有问题会失败
    except OSError as exc:
        raise RuntimeError(f"写入 Mosquitto 配置文件失败: {config_path} ({exc})") from exc

    # 尝试重载 / 重启 mosquitto
    def _run_cmd(cmd: List[str]) -> None:
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=30
            )
            # 如果返回码是负数，可能是被信号中断（如SIGINT），这是正常的（用户按Ctrl+C）
            if result.returncode != 0:
                # 检查是否是信号中断（负数退出码）
                if result.returncode < 0:
                    # 信号中断，可能是用户按Ctrl+C，这是正常的
                    # 不抛出异常，让上层处理
                    return
                error_msg = result.stderr.strip() or result.stdout.strip() or "未知错误"
                raise subprocess.CalledProcessError(
                    result.returncode,
                    cmd,
                    output=result.stdout,
                    stderr=result.stderr
                )
        except KeyboardInterrupt:
            # 用户中断，重新抛出让上层处理
            raise
        except subprocess.TimeoutExpired:
            # 超时，抛出异常
            raise RuntimeError(f"执行命令超时: {' '.join(cmd)}")

    if force_restart:
        # 强制使用 restart（完全重启）
        try:
            _run_cmd(["systemctl", "restart", "mosquitto"])
            # 等待服务启动
            time.sleep(2)
            # 验证服务是否成功启动
            result = subprocess.run(
                ["systemctl", "is-active", "mosquitto"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                # 获取详细错误信息
                journal_result = subprocess.run(
                    ["journalctl", "-u", "mosquitto.service", "-n", "10", "--no-pager"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                journal_output = journal_result.stdout if journal_result.returncode == 0 else ""
                raise RuntimeError(
                    f"Mosquitto 重启后服务未激活。\n"
                    f"配置文件: {config_path}\n"
                    f"请检查配置文件是否有语法错误。\n"
                    f"最近日志:\n{journal_output}"
                )
            return True  # 返回 True 表示使用了 restart
        except subprocess.CalledProcessError as exc:
            # 检查是否是信号中断（负数退出码，如SIGINT）
            if exc.returncode < 0:
                # 信号中断，可能是用户按Ctrl+C，重新抛出KeyboardInterrupt
                import signal
                if exc.returncode == -signal.SIGINT:
                    raise KeyboardInterrupt("用户中断（Ctrl+C）")
                else:
                    # 其他信号，可能是系统问题，抛出RuntimeError
                    raise RuntimeError(
                        f"重启 mosquitto 服务时被信号中断 (信号: {abs(exc.returncode)})\n"
                        f"配置文件: {config_path}\n"
                        f"这可能是正常的（如用户按Ctrl+C）"
                    ) from exc
            
            # 获取详细错误信息
            error_detail = exc.stderr or exc.stdout or str(exc)
            journal_result = subprocess.run(
                ["journalctl", "-u", "mosquitto.service", "-n", "10", "--no-pager"],
                capture_output=True,
                text=True,
                timeout=5
            )
            journal_output = journal_result.stdout if journal_result.returncode == 0 else ""
            raise RuntimeError(
                f"重启 mosquitto 服务失败 (退出码: {exc.returncode})\n"
                f"错误信息: {error_detail}\n"
                f"配置文件: {config_path}\n"
                f"请检查配置文件是否有语法错误。\n"
                f"最近日志:\n{journal_output}\n"
                f"提示: 可以手动检查配置文件: cat {config_path}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"重启 mosquitto 服务时发生异常: {exc}\n"
                f"配置文件: {config_path}"
            ) from exc
    else:
        # 优先使用 reload（重载配置，不断开连接）
        try:
            _run_cmd(["systemctl", "reload", "mosquitto"])
            return False  # 返回 False 表示使用了 reload
        except subprocess.CalledProcessError:
            # 若 reload 不支持或失败，回退到 restart
            print("[apply_knobs] reload 失败，尝试使用 restart...")
            try:
                _run_cmd(["systemctl", "restart", "mosquitto"])
                # 等待服务启动
                time.sleep(2)
                # 验证服务是否成功启动
                result = subprocess.run(
                    ["systemctl", "is-active", "mosquitto"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    journal_result = subprocess.run(
                        ["journalctl", "-u", "mosquitto.service", "-n", "10", "--no-pager"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    journal_output = journal_result.stdout if journal_result.returncode == 0 else ""
                    raise RuntimeError(
                        f"Mosquitto 重启后服务未激活。\n"
                        f"配置文件: {config_path}\n"
                        f"请检查配置文件是否有语法错误。\n"
                        f"最近日志:\n{journal_output}"
                    )
                return True  # 返回 True 表示使用了 restart
            except subprocess.CalledProcessError as exc:
                error_detail = exc.stderr or exc.stdout or str(exc)
                journal_result = subprocess.run(
                    ["journalctl", "-u", "mosquitto.service", "-n", "10", "--no-pager"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                journal_output = journal_result.stdout if journal_result.returncode == 0 else ""
                raise RuntimeError(
                    f"重载/重启 mosquitto 服务失败 (退出码: {exc.returncode})\n"
                    f"错误信息: {error_detail}\n"
                    f"配置文件: {config_path}\n"
                    f"请检查配置文件是否有语法错误。\n"
                    f"最近日志:\n{journal_output}\n"
                    f"提示: 可以手动检查配置文件: cat {config_path}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"重启 mosquitto 服务时发生异常: {exc}\n"
                    f"配置文件: {config_path}"
                ) from exc


