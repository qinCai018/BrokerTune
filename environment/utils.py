import json
import time
import threading
from typing import Dict, Optional, Tuple, List

import numpy as np

from .config import MQTTConfig, ProcConfig




"""
环境和采样相关的工具函数
    订阅 $SYS/#，解析 broker 指标
    读取 /proc/[pid]/stat 与 /proc/[pid]/status，获取 CPU/内存/ctxt
    拼接状态向量 s_t
"""

try:
    import paho.mqtt.client as mqtt
except ImportError:  # 允许项目先不安装，运行时再报错更清晰
    mqtt = None  # type: ignore


def _ensure_mqtt_available() -> None:
    if mqtt is None:
        raise RuntimeError(
            "paho-mqtt 未安装，请先在环境中安装：pip install paho-mqtt"
        )


class MQTTSampler:
    """
    简单的同步采样器：
    - 连接到 Mosquitto
    - 订阅 $SYS/# 等主题
    - 在一个小时间窗内收集若干条消息，解析为指标字典
    """

    def __init__(self, cfg: MQTTConfig):
        _ensure_mqtt_available()
        self._cfg = cfg
        self._client = mqtt.Client(client_id=cfg.client_id, clean_session=True)
        self._metrics: Dict[str, float] = {}
        self._metrics_ts: Dict[str, float] = {}
        self._rx_seq = 0
        self._lock = threading.Lock()
        self._connected = False  # 连接状态标志
        self._topic_prev: Dict[str, Tuple[float, float]] = {}
        self._topic_last: Dict[str, Tuple[float, float]] = {}
        self._topic_count: Dict[str, int] = {}

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        try:
            self._client.connect(cfg.host, cfg.port, keepalive=cfg.keepalive)
            self._client.loop_start()
            # 等待连接建立（最多等待 5 秒）
            import time
            for _ in range(50):  # 50 * 0.1 = 5 秒
                if self._connected:
                    break
                time.sleep(0.1)
            else:
                print(f"[MQTTSampler] 警告: 连接超时，可能无法连接到 MQTT broker")
        except Exception as e:
            print(f"[MQTTSampler] 连接失败: {e}")
            raise

    # ---------- MQTT 回调 ----------
    def _on_connect(self, client, userdata, flags, rc):
        if rc != 0:
            # 连接失败时打印错误信息
            error_messages = {
                1: "连接被拒绝 - 协议版本不正确",
                2: "连接被拒绝 - 客户端标识符无效",
                3: "连接被拒绝 - 服务器不可用",
                4: "连接被拒绝 - 用户名或密码错误",
                5: "连接被拒绝 - 未授权"
            }
            error_msg = error_messages.get(rc, f"未知错误 (rc={rc})")
            print(f"[MQTTSampler] 连接失败: {error_msg}")
            return
        self._connected = True
        for topic in self._cfg.topics:
            client.subscribe(topic)

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode("utf-8", errors="ignore").strip()
        value = _parse_numeric_payload(payload)
        if value is not None:
            now = time.time()
            with self._lock:
                self._rx_seq += 1
                self._metrics[topic] = value
                self._metrics_ts[topic] = now

                last = self._topic_last.get(topic)
                if last is not None:
                    last_value, last_time = last
                    if topic == "$SYS/broker/uptime" and value + 1e-6 < last_value:
                        # broker 重启导致 uptime 回退；清理历史，避免跨重启计算速率
                        self._topic_prev.pop(topic, None)
                        self._topic_last[topic] = (value, now)
                        self._topic_count[topic] = 1
                    else:
                        self._topic_prev[topic] = last
                        self._topic_last[topic] = (value, now)
                        self._topic_count[topic] = self._topic_count.get(topic, 0) + 1
                else:
                    self._topic_last[topic] = (value, now)
                    self._topic_count[topic] = 1

            if self._topic_count.get(topic, 0) <= 1:
                print(f"[MQTTSampler] 收到消息: {topic} = {value}")

    def _compute_rate_from_history(
        self, topic: str, min_interval_sec: float, min_samples: int
    ) -> Optional[float]:
        with self._lock:
            prev = self._topic_prev.get(topic)
            last = self._topic_last.get(topic)
            count = self._topic_count.get(topic, 0)
        if prev is None or last is None:
            return None
        if count < max(2, min_samples):
            return None
        prev_value, prev_time = prev
        last_value, last_time = last
        if last_time <= prev_time:
            return None
        if last_time - prev_time < min_interval_sec:
            return None
        delta = last_value - prev_value
        if delta < 0:
            return None
        return delta / (last_time - prev_time)

    # ---------- 公共接口 ----------
    def sample(self, timeout_sec: Optional[float] = None) -> Dict[str, float]:
        """
        在给定窗口内收集一轮 broker 指标。
        返回：{topic: value}，仅保留数值型 payload。
        """
        wait = timeout_sec if timeout_sec is not None else self._cfg.timeout_sec
        start_time = time.time()
        required_topics = list(getattr(self._cfg, "sample_wait_for_topics", []))

        def _is_required_topics_fresh() -> bool:
            if not required_topics:
                return True
            with self._lock:
                for topic in required_topics:
                    ts = self._metrics_ts.get(topic)
                    if ts is None or ts < start_time:
                        return False
            return True

        while time.time() - start_time < wait:
            ready = _is_required_topics_fresh()
            if ready and getattr(self._cfg, "sample_wait_for_derived_rate", True):
                derived = self._compute_rate_from_history(
                    "$SYS/broker/messages/received",
                    min_interval_sec=self._cfg.rate_min_interval_sec,
                    min_samples=self._cfg.rate_min_samples,
                )
                if derived is not None and derived > 0:
                    break
            if ready and not getattr(self._cfg, "sample_wait_for_derived_rate", True):
                break
            time.sleep(max(0.01, getattr(self._cfg, "sample_poll_interval_sec", 0.1)))

        with self._lock:
            metrics = dict(self._metrics)

        derived_rate = self._compute_rate_from_history(
            "$SYS/broker/messages/received",
            min_interval_sec=self._cfg.rate_min_interval_sec,
            min_samples=self._cfg.rate_min_samples,
        )
        if derived_rate is not None:
            metrics["$SYS/broker/messages/received_rate"] = derived_rate
        rate_1min_raw = metrics.get("$SYS/broker/load/messages/received/1min")
        if rate_1min_raw is not None and rate_1min_raw > 0:
            divisor = self._cfg.rate_1min_divisor
            if divisor and divisor > 0:
                metrics["$SYS/broker/load/messages/received/1min_per_sec"] = (
                    rate_1min_raw / divisor
                )
        print(f"[MQTTSampler] 采样完成，共收到 {len(metrics)} 条指标")
        return metrics

    def close(self):
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass


def _parse_numeric_payload(payload: str) -> Optional[float]:
    # 大部分 $SYS payload 是纯数字或简单字符串，这里做一个尽量鲁棒的解析
    try:
        return float(payload)
    except ValueError:
        pass

    # 处理 "123 seconds" 或 "123s" 这类 uptime 格式
    lowered = payload.lower()
    if "second" in lowered or lowered.endswith("s"):
        tokens = lowered.replace("seconds", "").replace("second", "").strip().split()
        if tokens:
            try:
                return float(tokens[0])
            except ValueError:
                pass

    # 尝试解析 JSON 里名为 value 的字段
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict) and "value" in obj:
            return float(obj["value"])
    except Exception:
        return None
    return None


def read_proc_metrics(cfg: ProcConfig) -> Tuple[float, float, float]:
    """
    从 /proc/<pid>/stat 与 /proc/<pid>/status 中提取：
    - cpu_ratio: 进程 CPU 占比（简单近似）
    - mem_ratio: RSS / mem_norm
    - ctxt_ratio: (voluntary_ctxt_switches + nonvoluntary) / ctxt_norm

    注意：这里的 CPU 估算比较粗糙，实际可结合 /proc/stat 做更精确的 delta 计算。
    """
    if cfg.pid <= 0:
        raise ValueError(f"ProcConfig.pid 未设置或非法 (当前值: {cfg.pid})，请正确配置 Mosquitto 进程 PID")

    # --- 内存 & 上下文切换 ---
    rss_bytes = 0.0
    ctxt_switches = 0.0

    status_path = f"/proc/{cfg.pid}/status"
    try:
        with open(status_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        # 单位通常是 kB
                        rss_kb = float(parts[1])
                        rss_bytes = rss_kb * 1024.0
                elif line.startswith("voluntary_ctxt_switches") or line.startswith(
                    "nonvoluntary_ctxt_switches"
                ):
                    parts = line.split()
                    if parts and parts[-1].isdigit():
                        ctxt_switches += float(parts[-1])
    except FileNotFoundError:
        # 进程已退出
        return 0.0, 0.0, 0.0

    # --- CPU 使用（粗略，读取 utime+stime）---
    stat_path = f"/proc/{cfg.pid}/stat"
    cpu_ticks = 0.0
    try:
        with open(stat_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        parts = content.split()
        # utime 在第 14 列，stime 在第 15 列（从 1 开始计）
        if len(parts) >= 15:
            utime = float(parts[13])
            stime = float(parts[14])
            cpu_ticks = utime + stime
    except FileNotFoundError:
        return 0.0, 0.0, 0.0

    # 简单归一化，实际可根据时间差再做 delta
    cpu_ratio = min(cpu_ticks / max(cfg.cpu_norm, 1.0), 1.0)
    mem_ratio = min(rss_bytes / max(cfg.mem_norm, 1.0), 1.0)
    ctxt_ratio = min(ctxt_switches / max(cfg.ctxt_norm, 1.0), 1.0)
    
    # 确保返回值是有效数值
    cpu_ratio = float(cpu_ratio) if not (np.isnan(cpu_ratio) or np.isinf(cpu_ratio)) else 0.0
    mem_ratio = float(mem_ratio) if not (np.isnan(mem_ratio) or np.isinf(mem_ratio)) else 0.0
    ctxt_ratio = float(ctxt_ratio) if not (np.isnan(ctxt_ratio) or np.isinf(ctxt_ratio)) else 0.0
    
    return cpu_ratio, mem_ratio, ctxt_ratio


def build_state_vector(
    broker_metrics: Dict[str, float],
    cpu_ratio: float,
    mem_ratio: float,
    ctxt_ratio: float,
    queue_depth: float = 0.0,
    throughput_history: Optional[List[float]] = None,
    latency_p50: float = 0.0,
    latency_p95: float = 0.0,
    latency_history: Optional[List[float]] = None,
    rate_1min_window_sec: float = 60.0,
) -> np.ndarray:
    """
    将 Mosquitto 运行指标 + 进程指标拼成状态向量 s_t。

    返回10维状态向量：
    - [0] broker 当前连接数（归一化）
    - [1] broker 消息速率（msg/s，归一化）
    - [2] CPU 使用占比
    - [3] RSS 内存占比
    - [4] 每秒上下文切换数占比
    - [5] P50 端到端延迟（ms，归一化）
    - [6] P95 端到端延迟（ms，归一化）
    - [7] 队列深度（归一化）
    - [8] 最近5步平均吞吐量（滑动窗口）
    - [9] 最近5步平均延迟（滑动窗口）
    """
    # 这些 key 可根据你的 broker 实际暴露的 $SYS 主题进行调整
    clients_connected = broker_metrics.get(
        "$SYS/broker/clients/connected", 0.0
    )
    
    # 优先使用采样窗口估算的速率，其次使用 1min 平均速率（换算为 msg/s）
    # 采样窗口速率与当前 action 更同步；1min 更平滑但滞后
    messages_rate_derived = broker_metrics.get(
        "$SYS/broker/messages/received_rate"
    )
    messages_rate_1min_per_sec = broker_metrics.get(
        "$SYS/broker/load/messages/received/1min_per_sec"
    )

    uptime_sec = broker_metrics.get("$SYS/broker/uptime")
    use_derived = (
        messages_rate_derived is not None and messages_rate_derived > 0
    )
    use_1min = (
        messages_rate_1min_per_sec is not None and messages_rate_1min_per_sec > 0
    )
    if uptime_sec is None:
        # 未拿到 uptime 时，不使用 1min 指标，避免在 broker 刚重启/采样不完整时误用滞后指标
        use_1min = False
    elif uptime_sec < rate_1min_window_sec:
        # Broker刚重启时1min指标不稳定，优先使用采样窗口估算
        use_1min = False

    if use_derived:
        messages_received = messages_rate_derived
    elif use_1min:
        messages_received = messages_rate_1min_per_sec
    else:
        messages_received = 0.0

    # 简单归一化，避免数量级差距过大
    clients_norm = clients_connected / 1000.0  # 假设 1000 连接为 1.0
    # 如果使用速率指标，已经是msg/s，直接归一化；如果使用累计值，需要除以时间
    msg_rate_norm = messages_received / 10000.0  # 假设 1w msg/s 为 1.0

    # 延迟归一化（假设 100ms 为 1.0）
    latency_p50_norm = latency_p50 / 100.0  # P50延迟归一化
    latency_p95_norm = latency_p95 / 100.0  # P95延迟归一化

    # 队列深度归一化（假设 1000 为 1.0）
    queue_depth_norm = queue_depth / 1000.0

    # 历史信息：滑动窗口平均值
    throughput_avg = np.mean(throughput_history) if throughput_history else msg_rate_norm
    latency_avg = np.mean(latency_history) if latency_history else latency_p50_norm

    # 归一化历史平均值（历史值本身已是归一化）
    throughput_avg_norm = throughput_avg
    latency_avg_norm = latency_avg

    state = np.array(
        [
            clients_norm,          # [0] 连接数归一化
            msg_rate_norm,         # [1] 消息速率归一化
            cpu_ratio,             # [2] CPU使用率
            mem_ratio,             # [3] 内存使用率
            ctxt_ratio,            # [4] 上下文切换率
            latency_p50_norm,      # [5] P50延迟归一化
            latency_p95_norm,      # [6] P95延迟归一化
            queue_depth_norm,      # [7] 队列深度归一化
            throughput_avg_norm,   # [8] 最近5步吞吐量平均
            latency_avg_norm,      # [9] 最近5步平均延迟
        ],
        dtype=np.float32,
    )
    
    # 确保所有值都是有效数值（替换NaN/Inf为0）
    state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return state
