import json
import time
from typing import Dict, Optional, Tuple

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
        self._connected = False  # 连接状态标志

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
            self._metrics[topic] = value
            # 调试：打印收到的消息（仅在开始时）
            if len(self._metrics) <= 3:  # 只打印前3条消息
                print(f"[MQTTSampler] 收到消息: {topic} = {value}")

    # ---------- 公共接口 ----------
    def sample(self, timeout_sec: Optional[float] = None) -> Dict[str, float]:
        """
        在给定窗口内收集一轮 broker 指标。
        返回：{topic: value}，仅保留数值型 payload。
        """
        self._metrics.clear()
        wait = timeout_sec if timeout_sec is not None else self._cfg.timeout_sec
        
        # 使用循环等待，每1秒检查一次是否收到消息，并打印进度
        import time
        start_time = time.time()
        check_interval = 1.0  # 每1秒检查一次
        last_count = 0
        
        while time.time() - start_time < wait:
            remaining = wait - (time.time() - start_time)
            sleep_time = min(check_interval, remaining)
            time.sleep(sleep_time)
            
            # 每1秒打印一次进度（如果收到新消息）
            current_count = len(self._metrics)
            if current_count != last_count:
                print(f"[MQTTSampler] 采样中... 已收到 {current_count} 条指标（剩余 {remaining:.1f}秒）")
                last_count = current_count
        
        metrics = dict(self._metrics)
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
) -> np.ndarray:
    """
    将 Mosquitto 运行指标 + 进程指标拼成状态向量 s_t。

    这里给出一个默认实现：
    - 连接数：从 "$SYS/broker/clients/connected" 获取，若不存在则为 0
    - 消息速率：从 "$SYS/broker/messages/received"、"$SYS/broker/messages/sent"
      等主题中择一（这里简单取 received）
    """
    # 这些 key 可根据你的 broker 实际暴露的 $SYS 主题进行调整
    clients_connected = broker_metrics.get(
        "$SYS/broker/clients/connected", 0.0
    )
    
    # 优先使用速率指标（1分钟内的消息速率），如果没有则使用累计值
    # $SYS/broker/load/messages/received/1min 是速率（msg/s）
    # $SYS/broker/messages/received 是累计值（Broker重启后重置）
    messages_rate_1min = broker_metrics.get(
        "$SYS/broker/load/messages/received/1min", 0.0
    )
    messages_received_total = broker_metrics.get(
        "$SYS/broker/messages/received", 0.0
    )
    
    # 如果速率指标可用，使用速率；否则使用累计值（不推荐，但作为后备）
    if messages_rate_1min > 0:
        messages_received = messages_rate_1min  # 已经是速率（msg/s）
    else:
        # 累计值，需要除以时间得到速率（这里假设采样间隔为1秒，但实际可能不准确）
        messages_received = messages_received_total  # 作为后备，但可能不准确

    # 简单归一化，避免数量级差距过大
    clients_norm = clients_connected / 1000.0  # 假设 1000 连接为 1.0
    # 如果使用速率指标，已经是msg/s，直接归一化；如果使用累计值，需要除以时间
    msg_rate_norm = messages_received / 10000.0  # 假设 1w msg/s 为 1.0

    state = np.array(
        [
            clients_norm,
            msg_rate_norm,
            cpu_ratio,
            mem_ratio,
            ctxt_ratio,
        ],
        dtype=np.float32,
    )
    
    # 确保所有值都是有效数值（替换NaN/Inf为0）
    state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return state

