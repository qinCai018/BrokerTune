import os
import time
import subprocess
from typing import Any, Dict, Tuple, Optional, List

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # 回退到旧版 gym（如果 gymnasium 不可用）
    import gym
    from gym import spaces
import numpy as np

from .config import EnvConfig
from .knobs import BrokerKnobSpace, apply_knobs
from .utils import MQTTSampler, build_state_vector, read_proc_metrics

"""
一个围绕 Mosquitto Broker 构建的 Gym 环境。
"""

class MosquittoBrokerEnv(gym.Env):
    """
    一个围绕 Mosquitto Broker 构建的 Gym 环境。

    采样流程：
    [1] 订阅 $SYS/# → 解析 Broker 负载
    [2] 读取 /proc/[pid]/stat → CPU
    [3] 读取 /proc/[pid]/status → 内存 / ctxt
    [4] 拼接状态向量 s_t
    """

    metadata = {"render.modes": []}

    def __init__(self, cfg: Optional[EnvConfig] = None, workload_manager: Optional[Any] = None):
        """
        初始化环境
        
        Args:
            cfg: 环境配置
            workload_manager: 工作负载管理器（可选），如果提供，Broker重启后将自动重启工作负载
        """
        super().__init__()
        self.cfg = cfg or EnvConfig()

        # 动作空间 / 状态空间
        self.knob_space = BrokerKnobSpace()
        assert (
            self.knob_space.action_dim == self.cfg.action_dim
        ), "EnvConfig.action_dim 必须与 BrokerKnobSpace.action_dim 一致"

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.cfg.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.state_dim,),
            dtype=np.float32,
        )

        # 采样器（延迟初始化，在reset时创建，确保Broker重启后重新连接）
        self._mqtt_sampler: Optional[MQTTSampler] = None

        # 工作负载管理器（可选）
        self._workload_manager = workload_manager

        self._last_broker_metrics: Dict[str, float] = {}
        self._last_broker_metrics_ts: float = 0.0

        self._step_count = 0
        self._last_state: Optional[np.ndarray] = None
        self._initial_state: Optional[np.ndarray] = None  # D_0: 初始状态性能
        self._last_applied_knobs: Optional[Dict[str, Any]] = None
        self._need_workload_restart = False  # 标志：是否需要重启工作负载

        # 历史状态跟踪（用于滑动窗口平均）
        self._throughput_history: List[float] = []  # 最近5步吞吐量
        self._latency_history: List[float] = []     # 最近5步延迟
        self._history_window = 5  # 滑动窗口大小
        self._last_probe_debug: Dict[str, Any] = {}
        self._last_latency_p50_ms: float = float(self.cfg.latency_fallback_p50_ms)
        self._last_latency_p95_ms: float = float(self.cfg.latency_fallback_p95_ms)
        self._last_queue_depth: float = 0.0
        self._consecutive_failures = 0
        self._restart_count = 0
        self._last_reward_components: Dict[str, float] = {}

    # ---------- 核心 Gym 接口 ----------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境：
        - 采集当前一轮状态作为 s_0，并保存为初始状态 D_0。
        
        Args:
            seed: 随机种子（可选，gymnasium兼容）
            options: 重置选项（可选，gymnasium兼容）
            
        Returns:
            observation: 初始状态
            info: 信息字典
        """
        # 如果采样器不存在或连接断开，重新创建
        if self._mqtt_sampler is None:
            try:
                self._mqtt_sampler = MQTTSampler(self.cfg.mqtt)
            except Exception as e:
                print(f"[MosquittoBrokerEnv] 警告: 创建MQTT采样器失败: {e}")
                print("[MosquittoBrokerEnv] 尝试重新创建...")
                time.sleep(1)
                self._mqtt_sampler = MQTTSampler(self.cfg.mqtt)
        else:
            # 检查连接是否有效
            if not self._mqtt_sampler._connected:
                print("[MosquittoBrokerEnv] MQTT采样器连接断开，重新创建...")
                try:
                    self._mqtt_sampler.close()
                except:
                    pass
                time.sleep(0.5)
                self._mqtt_sampler = MQTTSampler(self.cfg.mqtt)
        
        # 设置随机种子（如果提供）
        if seed is not None:
            np.random.seed(seed)
        
        # 在reset时应用默认配置，确保初始状态基于默认参数
        used_restart = False
        default_knobs = self.knob_space.get_default_knobs()
        if self.cfg.apply_default_on_reset:
            if self._last_applied_knobs != default_knobs:
                print("[MosquittoBrokerEnv] 应用默认Broker配置...")
                apply_knobs(default_knobs)
                used_restart = True  # 只要knob变化就视为重启
                self._last_applied_knobs = default_knobs.copy()
                if used_restart:
                    print("[MosquittoBrokerEnv] Broker已重启，等待稳定...")
                    self._wait_for_broker_ready(max_wait_sec=self.cfg.broker_restart_stable_sec)
                    # Broker 重启后，MQTT 采样器连接可能断开；这里强制重建，避免采样到旧连接的缓存
                    if self._mqtt_sampler is not None:
                        try:
                            self._mqtt_sampler.close()
                        except Exception:
                            pass
                        self._mqtt_sampler = None

                    # Broker 重启会断开所有客户端连接：确保工作负载恢复后再采样 baseline
                    if self._workload_manager is not None:
                        try:
                            if self._workload_manager.is_running():
                                self._workload_manager.stop()
                                time.sleep(1.0)
                            if getattr(self._workload_manager, "_last_config", None) is not None:
                                self._workload_manager.restart()
                                # 给工作负载一点时间建立连接并开始产生活动
                                time.sleep(max(2.0, float(self.cfg.baseline_retry_sleep_sec)))
                            else:
                                print("[MosquittoBrokerEnv] ⚠️  工作负载管理器无_last_config，无法在reset中自动重启工作负载")
                        except Exception as e:
                            print(f"[MosquittoBrokerEnv] ⚠️  reset中重启工作负载失败: {e}")
        
        self._step_count = 0
        self._need_workload_restart = False
        self._throughput_history = []
        self._latency_history = []
        self._consecutive_failures = 0
        
        # 在采样初始状态前，确保工作负载正在运行并稳定（仅在第一次reset时）
        if self._initial_state is None and self._workload_manager is not None:
            print("[MosquittoBrokerEnv] 准备采样初始状态，确保工作负载正在运行...")
            if not self._workload_manager.is_running():
                print("[MosquittoBrokerEnv] 工作负载未运行，启动工作负载...")
                try:
                    if self._workload_manager._last_config is not None:
                        self._workload_manager.restart()
                        print("[MosquittoBrokerEnv] 工作负载已启动，等待稳定运行（30秒）...")
                        time.sleep(30.0)
                        # 验证工作负载是否正常运行
                        if self._workload_manager.is_running():
                            print("[MosquittoBrokerEnv] ✅ 工作负载已稳定运行")
                        else:
                            print("[MosquittoBrokerEnv] ⚠️  警告：工作负载启动后未运行")
                    else:
                        print("[MosquittoBrokerEnv] ⚠️  警告：工作负载管理器没有保存的配置，无法启动")
                except Exception as e:
                    print(f"[MosquittoBrokerEnv] ⚠️  启动工作负载失败: {e}")
            else:
                print("[MosquittoBrokerEnv] ✅ 工作负载正在运行")
            
            # 等待$SYS主题发布（确保包含工作负载产生的消息）
            print("[MosquittoBrokerEnv] 等待$SYS主题发布（确保包含工作负载产生的消息，12秒）...")
            time.sleep(12.0)
        
        print("[MosquittoBrokerEnv] 开始采样初始状态...")
        state = None
        baseline_ok = False
        last_candidate: Optional[np.ndarray] = None
        for attempt in range(max(1, int(self.cfg.baseline_max_attempts))):
            candidate = self._sample_state()

            # 验证状态有效性
            if np.any(np.isnan(candidate)) or np.any(np.isinf(candidate)):
                print("[MosquittoBrokerEnv] 警告: reset时检测到无效状态值（NaN/Inf），使用零状态")
                candidate = np.zeros(self.cfg.state_dim, dtype=np.float32)

            # 限制状态值范围
            candidate = np.clip(candidate, -1e6, 1e6)
            last_candidate = candidate

            clients_norm = float(candidate[0])
            throughput_norm = float(candidate[1])
            if (
                clients_norm >= float(self.cfg.baseline_min_clients_norm)
                and throughput_norm >= float(self.cfg.baseline_min_throughput)
            ):
                baseline_ok = True
                state = candidate
                break

            if attempt < int(self.cfg.baseline_max_attempts) - 1:
                print(
                    "[MosquittoBrokerEnv] ⚠️  baseline采样过低，重试 "
                    f"{attempt + 1}/{self.cfg.baseline_max_attempts}："
                    f"clients_norm={clients_norm:.6f}, throughput_norm={throughput_norm:.6f}"
                )
                time.sleep(float(self.cfg.baseline_retry_sleep_sec))

        if state is None:
            state = last_candidate if last_candidate is not None else np.zeros(self.cfg.state_dim, dtype=np.float32)
        print("[MosquittoBrokerEnv] 初始状态采样完成")
        
        # 根据配置决定是否每个 episode 更新基线
        if self.cfg.baseline_per_episode or self._initial_state is None:
            if baseline_ok or self._initial_state is None:
                self._initial_state = state.copy()
            else:
                print("[MosquittoBrokerEnv] ⚠️  baseline不达标，保留上一episode的初始基线以避免奖励失真")
            if hasattr(self, "_initial_throughput_logged"):
                delattr(self, "_initial_throughput_logged")
            initial_throughput = self._extract_throughput(self._initial_state)
            print(f"[MosquittoBrokerEnv] ✅ 已设置episode初始吞吐量: {initial_throughput:.6f}")
        
        self._last_state = state
        
        # gymnasium兼容：返回 (observation, info) 元组
        info: Dict[str, Any] = {
            "step": self._step_count,
            "episode_initial_throughput": float(state[1]) if len(state) > 1 else 0.0,
            "episode_initial_latency_p50": float(state[5]) if len(state) > 5 else 0.0,
            "restart_count": self._restart_count,
        }
        return state, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        单步交互：
        - 将连续动作向量映射为 broker knobs，并应用到系统
        - 根据是否重启 Broker，等待相应时间让系统稳定
        - 再采一轮状态，计算 reward
        
        Returns:
            observation: 新状态
            reward: 奖励值
            terminated: 是否终止（episode结束）
            truncated: 是否截断（时间限制等）
            info: 信息字典
        """
        # 验证动作在有效范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action.astype(np.float32)
        
        self._step_count += 1

        # 1. 解码并应用 knobs
        try:
            knobs = self.knob_space.decode_action(action)
        except Exception as exc:
            return self._make_failure_transition(
                reason="decode_action_failed",
                error=exc,
                knobs={},
            )
        used_restart = False
        if self._last_applied_knobs == knobs:
            if self._step_count <= 3 or self._step_count % 10 == 0:
                print("[MosquittoBrokerEnv] 配置未变化，跳过应用与重启")
        else:
            try:
                apply_knobs(knobs)
            except Exception as exc:
                return self._make_failure_transition(
                    reason="apply_knobs_failed",
                    error=exc,
                    knobs=knobs,
                )
            used_restart = True  # 只要knob变化就视为重启
            self._last_applied_knobs = knobs.copy()
            self._restart_count += 1
        
        # 记录Broker重启信息（用于工作负载健康检查）
        # 注意：Broker重启会导致所有MQTT连接断开，包括工作负载
        if used_restart:
            # 将Broker重启信息存储到环境属性中，供callback访问
            if not hasattr(self, '_broker_restart_steps'):
                self._broker_restart_steps = []
            self._broker_restart_steps.append(self._step_count)
            
            # 设置标志，通知callback需要立即重启工作负载
            self._need_workload_restart = True
            
            # Broker重启后，MQTT采样器连接也会断开
            # 注意：采样器的重新创建会在等待工作负载稳定运行30秒之后进行（在 _sample_state() 中）
            # 这里先关闭旧连接，但不立即重新创建，等待工作负载稳定后再创建

        # 2. 等待系统稳定
        # 如果使用了 restart（完全重启），需要等待更长时间并检查 Broker 是否启动成功
        # 如果使用了 reload（重载配置），等待时间可以较短
        if used_restart:
            stable_wait_sec = self.cfg.broker_restart_stable_sec
            if self._step_count <= 3 or self._step_count % 10 == 0:  # 只在开始几步或每10步打印一次
                print(f"[MosquittoBrokerEnv] Broker 已重启，等待系统稳定...")
            # 动态检查 Broker 是否启动成功，最多等待 stable_wait_sec 秒
            self._wait_for_broker_ready(max_wait_sec=stable_wait_sec)
            
            # Broker重启后，工作负载也会断开，需要立即重启工作负载并等待稳定运行
            # 如果提供了工作负载管理器，在这里立即重启；否则依赖callback重启
            # 注意：先重启工作负载，再等待$SYS主题发布，这样$SYS主题会包含工作负载产生的消息
            if self._workload_manager is not None:
                if self._step_count <= 3 or self._step_count % 10 == 0:
                    print(f"[MosquittoBrokerEnv] Broker重启就绪，立即重启工作负载...")
                
                try:
                    # 检查工作负载是否还在运行（Broker重启后应该已断开）
                    if self._workload_manager.is_running():
                        if self._step_count <= 3 or self._step_count % 10 == 0:
                            print(f"[MosquittoBrokerEnv] 停止旧工作负载进程...")
                        self._workload_manager.stop()
                        time.sleep(1.0)  # 等待进程完全停止
                    
                    # 重启工作负载（使用保存的配置）
                    if self._workload_manager._last_config is not None:
                        if self._step_count <= 3 or self._step_count % 10 == 0:
                            config = self._workload_manager._last_config
                            print(f"[MosquittoBrokerEnv] 重启工作负载（使用原配置：{config.num_publishers}发布者，{config.num_subscribers}订阅者，主题'{config.topic}'，QoS={config.qos}，间隔={config.publisher_interval_ms}ms，消息大小={config.message_size}B）...")
                        self._workload_manager.restart()
                        
                        # 等待工作负载稳定运行30秒
                        if self._step_count <= 3 or self._step_count % 10 == 0:
                            print(f"[MosquittoBrokerEnv] 工作负载重启成功，等待稳定运行（30秒）...")
                        time.sleep(30.0)
                        
                        # 验证工作负载是否正常运行
                        if self._workload_manager.is_running():
                            if self._step_count <= 3 or self._step_count % 10 == 0:
                                print(f"[MosquittoBrokerEnv] ✅ 工作负载已稳定运行（进程数: {len(self._workload_manager._processes)}）")
                                # 可选：验证消息发送
                                if self._workload_manager._last_config.num_publishers > 0:
                                    if self._workload_manager._verify_messages_sending(
                                        self._workload_manager._last_config.topic, 
                                        timeout_sec=5.0
                                    ):
                                        if self._step_count <= 3 or self._step_count % 10 == 0:
                                            print(f"[MosquittoBrokerEnv] ✅ 验证成功：工作负载正在发送消息到主题 '{self._workload_manager._last_config.topic}'")
                        else:
                            print(f"[MosquittoBrokerEnv] ⚠️  警告：工作负载重启后未运行")
                    else:
                        print(f"[MosquittoBrokerEnv] ⚠️  警告：工作负载管理器没有保存的配置，无法重启")
                        if self._step_count <= 3 or self._step_count % 10 == 0:
                            print(f"[MosquittoBrokerEnv] 提示：工作负载健康检查callback将尝试重启工作负载")
                        time.sleep(30.0)  # 仍然等待，给callback时间重启
                except Exception as e:
                    print(f"[MosquittoBrokerEnv] ❌ 重启工作负载失败: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"[MosquittoBrokerEnv] 提示：工作负载健康检查callback将尝试重启工作负载")
                    time.sleep(30.0)  # 仍然等待，给callback时间重启
            else:
                # 如果没有提供工作负载管理器，依赖callback重启
                if self._step_count <= 3 or self._step_count % 10 == 0:
                    print(f"[MosquittoBrokerEnv] Broker重启完成，等待工作负载重启并稳定运行（30秒）...")
                    print(f"[MosquittoBrokerEnv] 提示：工作负载健康检查callback将立即重启工作负载")
                time.sleep(30.0)  # 等待工作负载稳定运行30秒
            
            # 工作负载稳定后，等待$SYS主题发布
            # 注意：在重启工作负载之后等待$SYS主题发布，这样$SYS主题会包含工作负载产生的消息
            # Broker重启后，需要等待sys_interval时间才会发布第一个$SYS消息
            # 通常sys_interval=10秒，所以额外等待12秒确保能收到$SYS消息
            if self._step_count <= 3 or self._step_count % 10 == 0:
                print(f"[MosquittoBrokerEnv] 工作负载已稳定，等待$SYS主题发布（Broker重启后需要sys_interval时间，通常10秒）...")
            time.sleep(12.0)  # 等待sys_interval时间（通常10秒）+ 2秒缓冲
            
            # 清除工作负载重启标志
            self._need_workload_restart = False
            
            # 等待工作负载稳定后，确保MQTT采样器已准备好（如果需要重新创建）
            # 注意：MQTT采样器的创建和连接验证会在 _sample_state() 中进行
            # 但这里先标记需要重新创建（如果Broker重启导致连接断开）
            if self._mqtt_sampler is not None:
                try:
                    self._mqtt_sampler.close()
                except:
                    pass
                self._mqtt_sampler = None  # 标记为需要重新创建
        else:
            stable_wait_sec = self.cfg.step_interval_sec
            if stable_wait_sec > 0:
                if self._step_count <= 3 or self._step_count % 10 == 0:
                    print(f"[MosquittoBrokerEnv] 配置未重启，等待 {stable_wait_sec} 秒让系统稳定...")
                time.sleep(stable_wait_sec)

        # 3. 采样新状态（在等待工作负载稳定运行30秒之后进行）
        # 此时工作负载应该已经稳定运行，可以采集准确的指标
        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 开始采样新状态（步数: {self._step_count}）...")
        try:
            next_state = self._sample_state()
        except Exception as exc:
            return self._make_failure_transition(
                reason="sample_state_failed",
                error=exc,
                knobs=knobs,
            )

        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 新状态采样完成")

        self._consecutive_failures = 0

        # 更新历史记录（用于滑动窗口）
        throughput = float(next_state[1])  # msg_rate_norm
        latency = float(next_state[5])    # latency_p50_norm

        self._throughput_history.append(throughput)
        self._latency_history.append(latency)

        # 保持历史记录在窗口大小内
        if len(self._throughput_history) > self._history_window:
            self._throughput_history.pop(0)
        if len(self._latency_history) > self._history_window:
            self._latency_history.pop(0)
        
        # 验证状态有效性（防止NaN/Inf）
        if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
            print(f"[MosquittoBrokerEnv] 警告: 检测到无效状态值（NaN/Inf），使用零状态")
            next_state = np.zeros_like(next_state, dtype=np.float32)
        
        # 限制状态值范围，防止极端值
        next_state = np.clip(next_state, -1e6, 1e6)

        # 4. 计算奖励（示例逻辑，可按需改写）
        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 开始计算奖励（prev_state={self._last_state is not None}, next_state={next_state is not None}）...")
        try:
            reward = self._compute_reward(
                prev_state=self._last_state,
                next_state=next_state,
            )
        except Exception as exc:
            return self._make_failure_transition(
                reason="reward_compute_failed",
                error=exc,
                knobs=knobs,
            )
        
        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 奖励计算完成: {reward:.6f}")
        
        # 验证奖励有效性
        if np.isnan(reward) or np.isinf(reward):
            print(f"[MosquittoBrokerEnv] 警告: 检测到无效奖励值（NaN/Inf），使用0.0")
            reward = 0.0
        
        # 限制奖励范围，防止极端值
        reward = np.clip(reward, -1e6, 1e6)

        # gymnasium v0.26+ 格式：返回 (obs, reward, terminated, truncated, info)
        terminated = self._step_count >= self.cfg.max_steps  # episode正常结束
        truncated = False  # 没有截断条件（可以后续添加）
        probe_debug = dict(self._last_probe_debug)
        latency_source = "probe" if probe_debug.get("samples", 0) > 0 else "fallback"
        throughput_msg_per_sec = float(next_state[1]) * 10000.0
        latency_p50_ms = float(next_state[5]) * 100.0
        latency_p95_ms = float(next_state[6]) * 100.0
        info: Dict[str, Any] = {
            "knobs": knobs,
            "step": self._step_count,
            "throughput_norm": float(next_state[1]),
            "throughput_msg_per_sec": throughput_msg_per_sec,
            "latency_p50_ms": latency_p50_ms,
            "latency_p95_ms": latency_p95_ms,
            "queue_depth_norm": float(next_state[7]) if len(next_state) > 7 else 0.0,
            "cpu_ratio": float(next_state[2]) if len(next_state) > 2 else 0.0,
            "mem_ratio": float(next_state[3]) if len(next_state) > 3 else 0.0,
            "ctxt_ratio": float(next_state[4]) if len(next_state) > 4 else 0.0,
            "restart_count": self._restart_count,
            "consecutive_failures": self._consecutive_failures,
            "reward_components": dict(self._last_reward_components),
            "latency_source": latency_source,
            "latency_probe_connected": bool(probe_debug.get("connected", False)),
            "latency_probe_samples": int(probe_debug.get("samples", 0)),
            "latency_probe_min": float(probe_debug.get("min", 0.0)),
            "latency_probe_max": float(probe_debug.get("max", 0.0)),
            "latency_probe_p50": float(probe_debug.get("p50", 0.0)),
            "latency_probe_p95": float(probe_debug.get("p95", 0.0)),
            "latency_probe_topic": probe_debug.get("topic", ""),
        }

        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] step() 完成: reward={reward:.6f}, terminated={terminated}, truncated={truncated}")

        self._last_state = next_state
        return next_state, float(reward), bool(terminated), bool(truncated), info

    def _make_failure_transition(
        self,
        reason: str,
        error: Exception,
        knobs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        将关键路径异常转换为可学习的失败转移，避免训练进程直接崩溃。
        """
        self._consecutive_failures += 1
        fallback_state = (
            self._last_state.copy()
            if self._last_state is not None
            else np.zeros(self.cfg.state_dim, dtype=np.float32)
        )
        terminated = self._step_count >= self.cfg.max_steps
        truncated = self._consecutive_failures >= int(self.cfg.max_consecutive_failures)
        reward = float(self.cfg.failed_step_penalty)
        info: Dict[str, Any] = {
            "step": self._step_count,
            "knobs": knobs or {},
            "error_reason": reason,
            "error": str(error),
            "consecutive_failures": self._consecutive_failures,
            "restart_count": self._restart_count,
            "latency_source": "error",
            "throughput_norm": float(fallback_state[1]) if len(fallback_state) > 1 else 0.0,
            "latency_p50_ms": float(fallback_state[5]) * 100.0 if len(fallback_state) > 5 else 0.0,
        }
        print(f"[MosquittoBrokerEnv] ❌ {reason}: {error}")
        if truncated:
            print(
                f"[MosquittoBrokerEnv] 连续失败达到阈值({self.cfg.max_consecutive_failures})，截断当前 episode"
            )
        return fallback_state, reward, bool(terminated), bool(truncated), info

    def render(self, mode: str = "human"):
        # 可以添加一些简单的日志输出或可视化
        return None

    def close(self):
        if self._mqtt_sampler is not None:
            try:
                self._mqtt_sampler.close()
            except:
                pass

    # ---------- 内部工具 ----------
    def _sample_state(self) -> np.ndarray:
        # 确保采样器存在且连接
        if self._mqtt_sampler is None or not self._mqtt_sampler._connected:
            print("[MosquittoBrokerEnv] MQTT采样器未连接，重新创建...")
            try:
                if self._mqtt_sampler:
                    self._mqtt_sampler.close()
            except:
                pass
            time.sleep(0.5)
            
            # 创建采样器，最多重试3次
            max_retries = 3
            for retry in range(max_retries):
                try:
                    self._mqtt_sampler = MQTTSampler(self.cfg.mqtt)
                    # 等待连接建立（最多等待5秒）
                    for _ in range(50):  # 50 * 0.1 = 5秒
                        if self._mqtt_sampler._connected:
                            if self._step_count <= 3 or self._step_count % 10 == 0:
                                print(f"[MosquittoBrokerEnv] ✅ MQTT采样器连接成功")
                            break
                        time.sleep(0.1)
                    else:
                        # 连接超时
                        if retry < max_retries - 1:
                            print(f"[MosquittoBrokerEnv] ⚠️  MQTT采样器连接超时，重试 {retry + 1}/{max_retries}...")
                            try:
                                self._mqtt_sampler.close()
                            except:
                                pass
                            self._mqtt_sampler = None
                            time.sleep(1.0)
                            continue
                        else:
                            print(f"[MosquittoBrokerEnv] ❌ MQTT采样器连接失败（已重试{max_retries}次）")
                            raise RuntimeError("MQTT采样器连接失败")
                    
                    # 连接成功，退出重试循环
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"[MosquittoBrokerEnv] ⚠️  创建MQTT采样器失败: {e}，重试 {retry + 1}/{max_retries}...")
                        time.sleep(1.0)
                        continue
                    else:
                        print(f"[MosquittoBrokerEnv] ❌ 创建MQTT采样器失败（已重试{max_retries}次）: {e}")
                        raise
        
        # Broker 指标
        # 如果Broker刚重启，可能需要等待sys_interval时间才能收到$SYS消息
        # 使用较长的采样时间确保能收到消息
        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 开始采样Broker指标（超时: {self.cfg.mqtt.timeout_sec}秒）...")
        
        broker_metrics = self._mqtt_sampler.sample(timeout_sec=self.cfg.mqtt.timeout_sec)
        self._last_broker_metrics = broker_metrics.copy()
        self._last_broker_metrics_ts = time.time()
        
        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 采样完成，收到 {len(broker_metrics)} 条指标")
            if len(broker_metrics) > 0:
                # 显示前几个指标作为示例
                sample_keys = list(broker_metrics.keys())[:3]
                for key in sample_keys:
                    print(f"[MosquittoBrokerEnv]   {key}: {broker_metrics[key]}")
        
        # 如果未收到任何指标，打印警告
        if len(broker_metrics) == 0:
            if self._step_count <= 3 or self._step_count % 20 == 0:  # 只在开始几步或每20步打印一次
                print(f"[MosquittoBrokerEnv] 警告: 未收到任何$SYS主题消息（步数: {self._step_count}）")
                print("[MosquittoBrokerEnv] 可能原因:")
                print("  1. Broker未配置sys_interval（不发布$SYS主题）")
                print("  2. Broker刚重启，$SYS主题还未发布（需要等待sys_interval时间）")
                print("  3. 采样时间太短（当前: {:.1f}秒）".format(self.cfg.mqtt.timeout_sec))
                print("[MosquittoBrokerEnv] 建议: 检查Broker配置是否有 'sys_interval 1'（或更小）")
        
        # 进程指标
        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 读取进程指标...")
        try:
            cpu_ratio, mem_ratio, ctxt_ratio = read_proc_metrics(self.cfg.proc)
        except Exception as exc:
            cpu_ratio, mem_ratio, ctxt_ratio = 0.0, 0.0, 0.0
            print(f"[MosquittoBrokerEnv] ⚠️  读取进程指标失败，回退为0: {exc}")
        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 进程指标: CPU={cpu_ratio:.4f}, MEM={mem_ratio:.4f}, CTXT={ctxt_ratio:.4f}")
        # 拼接状态向量
        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 构建状态向量...")

        # 获取延迟和队列深度指标
        probe_debug: Dict[str, Any] = {}
        latency_p50 = float(self.cfg.latency_fallback_p50_ms)
        latency_p95 = float(self.cfg.latency_fallback_p95_ms)
        if (
            self.cfg.enable_latency_probe
            and self._workload_manager is not None
            and hasattr(self._workload_manager, "get_latency_probe_debug")
        ):
            try:
                probe_debug = self._workload_manager.get_latency_probe_debug()
            except Exception as exc:
                print(f"[MosquittoBrokerEnv] ⚠️  获取延迟探测信息失败: {exc}")
                probe_debug = {}

        probe_samples = int(probe_debug.get("samples", 0))
        if probe_samples > 0:
            latency_p50 = float(probe_debug.get("p50", latency_p50))
            latency_p95 = float(probe_debug.get("p95", latency_p95))
        elif self._last_latency_p50_ms > 0 and self._last_latency_p95_ms > 0:
            latency_p50 = float(self._last_latency_p50_ms)
            latency_p95 = float(self._last_latency_p95_ms)

        queue_depth = self._extract_queue_depth(broker_metrics)

        self._last_probe_debug = probe_debug
        self._last_latency_p50_ms = float(latency_p50)
        self._last_latency_p95_ms = float(latency_p95)
        self._last_queue_depth = float(queue_depth)

        state = build_state_vector(
            broker_metrics,
            cpu_ratio,
            mem_ratio,
            ctxt_ratio,
            queue_depth=queue_depth,
            throughput_history=self._throughput_history,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_history=self._latency_history,
            rate_1min_window_sec=self.cfg.mqtt.rate_1min_window_sec,
        )
        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 状态向量构建完成: {state}")
            print(f"[MosquittoBrokerEnv] _sample_state() 完成，返回状态")
        return state

    def _compute_reward(
        self,
        prev_state: Optional[np.ndarray],
        next_state: np.ndarray,
    ) -> float:
        """
        组合奖励：
        - 吞吐量提升（越高越好）
        - 时延降低（越低越好）
        """
        eps = 1e-6
        denom_floor = max(float(self.cfg.baseline_min_throughput), eps)
        latency_floor = max(float(self.cfg.reward_latency_floor_norm), eps)

        current_throughput = self._extract_throughput(next_state)
        prev_throughput = self._extract_throughput(prev_state) if prev_state is not None else current_throughput

        current_latency = self._extract_latency(next_state)
        prev_latency = self._extract_latency(prev_state) if prev_state is not None else current_latency

        initial_throughput = current_throughput
        initial_latency = current_latency
        if self._initial_state is not None:
            initial_throughput = self._extract_throughput(self._initial_state)
            initial_latency = self._extract_latency(self._initial_state)
            if not hasattr(self, '_initial_throughput_logged'):
                print(f"[Reward] 📌 episode初始吞吐量: {initial_throughput:.6f}")
                self._initial_throughput_logged = True

        if self._throughput_history:
            avg_throughput = float(np.mean(self._throughput_history))
        else:
            avg_throughput = current_throughput
        if self._latency_history:
            avg_latency = float(np.mean(self._latency_history))
        else:
            avg_latency = current_latency

        if initial_throughput < denom_floor:
            delta_throughput_base = 0.0
        else:
            delta_throughput_base = (avg_throughput - initial_throughput) / initial_throughput

        denom_step = max(prev_throughput, denom_floor)
        delta_throughput_step = (current_throughput - prev_throughput) / denom_step

        if initial_latency < latency_floor:
            delta_latency_base = 0.0
        else:
            delta_latency_base = (initial_latency - avg_latency) / initial_latency
        delta_latency_step = (prev_latency - current_latency) / max(prev_latency, latency_floor)

        if self.cfg.reward_use_tanh:
            delta_throughput_base = np.tanh(delta_throughput_base)
            delta_throughput_step = np.tanh(delta_throughput_step)
            delta_latency_base = np.tanh(delta_latency_base)
            delta_latency_step = np.tanh(delta_latency_step)
        else:
            delta_throughput_base = np.clip(
                delta_throughput_base, -self.cfg.reward_delta_clip, self.cfg.reward_delta_clip
            )
            delta_throughput_step = np.clip(
                delta_throughput_step, -self.cfg.reward_delta_clip, self.cfg.reward_delta_clip
            )
            delta_latency_base = np.clip(
                delta_latency_base, -self.cfg.reward_delta_clip, self.cfg.reward_delta_clip
            )
            delta_latency_step = np.clip(
                delta_latency_step, -self.cfg.reward_delta_clip, self.cfg.reward_delta_clip
            )

        reward = self.cfg.reward_scale * (
            self.cfg.reward_weight_base * delta_throughput_base
            + self.cfg.reward_weight_step * delta_throughput_step
            + self.cfg.reward_weight_latency_base * delta_latency_base
            + self.cfg.reward_weight_latency_step * delta_latency_step
        )

        reward = float(reward)
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        reward = np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip)
        self._last_reward_components = {
            "throughput_base": float(delta_throughput_base),
            "throughput_step": float(delta_throughput_step),
            "latency_base": float(delta_latency_base),
            "latency_step": float(delta_latency_step),
            "reward": float(reward),
        }

        # 打印奖励信息（调试用）
        if self._step_count <= 3 or self._step_count % 20 == 0:
            prev_throughput_str = f"{prev_throughput:.6f}" if prev_state is not None else "N/A"
            initial_throughput_str = f"{initial_throughput:.6f}" if self._initial_state is not None else "N/A"
            prev_latency_str = f"{prev_latency:.6f}" if prev_state is not None else "N/A"
            initial_latency_str = f"{initial_latency:.6f}" if self._initial_state is not None else "N/A"
            reward_type = "正向" if reward > 0 else ("负向" if reward < 0 else "零")
            print(f"[Reward] 当前吞吐量: {current_throughput:.6f}, "
                  f"平均吞吐量: {avg_throughput:.6f}, "
                  f"上一时刻吞吐量: {prev_throughput_str}, "
                  f"初始吞吐量: {initial_throughput_str}, "
                  f"当前延迟: {current_latency:.6f}, "
                  f"平均延迟: {avg_latency:.6f}, "
                  f"上一时刻延迟: {prev_latency_str}, "
                  f"初始延迟: {initial_latency_str}, "
                  f"Δ_tp_base: {delta_throughput_base:+.6f}, "
                  f"Δ_tp_step: {delta_throughput_step:+.6f}, "
                  f"Δ_lat_base: {delta_latency_base:+.6f}, "
                  f"Δ_lat_step: {delta_latency_step:+.6f}, "
                  f"奖励类型: {reward_type}, "
                  f"总奖励: {reward:.6f}")

        return reward

    def get_last_broker_metrics(self) -> Dict[str, float]:
        """
        返回最近一次采样的 broker 指标。
        """
        return dict(self._last_broker_metrics)

    def _extract_queue_depth(self, broker_metrics: Dict[str, float]) -> float:
        """
        从 broker 指标中提取队列深度。
        若多个候选指标都存在，优先使用消息队列长度相关字段。
        """
        candidate_keys = [
            "$SYS/broker/store/messages/count",
            "$SYS/broker/messages/stored",
            "$SYS/broker/retained messages/count",
            "$SYS/broker/heap/messages",
        ]
        for key in candidate_keys:
            value = broker_metrics.get(key)
            if value is not None and value >= 0:
                return float(value)
        return 0.0
    
    def _extract_throughput(self, state: np.ndarray) -> float:
        """
        从状态向量中提取吞吐量指标。

        Args:
            state: 状态向量

        Returns:
            吞吐量值（归一化后的）
        """
        return float(state[1])  # msg_rate_norm

    def _extract_latency(self, state: np.ndarray) -> float:
        """
        从状态向量中提取延迟指标。

        Args:
            state: 状态向量

        Returns:
            延迟值（归一化后的）
        """
        return float(state[5])  # latency_p50_norm

    def _extract_performance_metric(self, state: np.ndarray) -> float:
        """
        从状态向量中提取综合性能指标 D（向后兼容）。

        现在使用吞吐量和延迟的组合作为性能指标：
        performance = throughput - latency_penalty

        Args:
            state: 状态向量

        Returns:
            综合性能指标值（归一化后的）
        """
        throughput = self._extract_throughput(state)
        latency = self._extract_latency(state)

        # 性能指标 = 吞吐量 - 延迟惩罚
        performance = throughput - 0.5 * latency  # 延迟权重可以调优

        return performance
    
    def _wait_for_broker_ready(self, max_wait_sec: float = 30.0, check_interval_sec: float = 1.0) -> None:
        """
        等待 Broker 启动并准备就绪。
        
        通过检查 systemctl status、进程是否存在、以及端口是否监听来确认 Broker 是否启动成功。
        如果 Broker 在 max_wait_sec 内未就绪，会继续等待但不会抛出异常
        （因为可能只是启动较慢）。
        
        重要：Broker 重启后 PID 可能会改变，此方法会自动更新 PID。
        
        Args:
            max_wait_sec: 最大等待时间（秒）
            check_interval_sec: 检查间隔（秒）
        """
        start_time = time.time()
        elapsed = 0.0
        
        def _check_port_listening(port: int = 1883) -> bool:
            """检查端口是否监听"""
            try:
                # 尝试使用 netstat 或 ss 检查端口
                for cmd in [["netstat", "-tln"], ["ss", "-tln"]]:
                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if result.returncode == 0:
                            if f":{port}" in result.stdout or f"*:{port}" in result.stdout:
                                return True
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        continue
                return False
            except Exception:
                return False
        
        while elapsed < max_wait_sec:
            # 检查 systemctl 状态
            try:
                result = subprocess.run(
                    ["systemctl", "is-active", "mosquitto"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0 and result.stdout.strip() == "active":
                    # Broker 服务已激活，检查端口是否监听
                    port_ready = _check_port_listening(1883)
                    
                    # 尝试获取新的 PID
                    try:
                        pid_result = subprocess.run(
                            ["pgrep", "-o", "mosquitto"],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if pid_result.returncode == 0 and pid_result.stdout.strip():
                            new_pid = int(pid_result.stdout.strip())
                            # 检查新 PID 对应的进程是否存在
                            if os.path.exists(f"/proc/{new_pid}"):
                                # 更新 PID（如果改变了）
                                if self.cfg.proc.pid != new_pid:
                                    old_pid = self.cfg.proc.pid
                                    self.cfg.proc.pid = new_pid
                                    os.environ["MOSQUITTO_PID"] = str(new_pid)
                                    if elapsed > 0:
                                        port_status = "端口已监听" if port_ready else "端口未监听"
                                        print(f"[MosquittoBrokerEnv] Broker 已就绪（PID: {old_pid} -> {new_pid}，{port_status}，等待了 {elapsed:.1f} 秒）")
                                    else:
                                        print(f"[MosquittoBrokerEnv] Broker PID 已更新: {old_pid} -> {new_pid}")
                                else:
                                    if elapsed > 0:
                                        port_status = "端口已监听" if port_ready else "端口未监听"
                                        print(f"[MosquittoBrokerEnv] Broker 已就绪（{port_status}，等待了 {elapsed:.1f} 秒）")
                                
                                # 如果端口已监听，认为Broker完全就绪
                                if port_ready:
                                    return
                    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
                        pass
                    
                    # 如果无法获取新 PID，但服务已激活且端口监听，认为就绪
                    if port_ready:
                        pid = self.cfg.proc.pid
                        if pid > 0 and os.path.exists(f"/proc/{pid}"):
                            if elapsed > 0:
                                print(f"[MosquittoBrokerEnv] Broker 已就绪（端口已监听，等待了 {elapsed:.1f} 秒）")
                            return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            time.sleep(check_interval_sec)
            elapsed = time.time() - start_time
        
        # 如果超时仍未就绪，打印警告但继续执行
        if elapsed >= max_wait_sec:
            port_status = "端口已监听" if _check_port_listening(1883) else "端口未监听"
            print(f"[MosquittoBrokerEnv] 警告: Broker 在 {max_wait_sec} 秒内可能未完全就绪（{port_status}），继续执行...")
