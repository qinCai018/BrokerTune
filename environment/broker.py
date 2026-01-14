import os
import time
import subprocess
from typing import Any, Dict, Tuple, Optional

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

        self._step_count = 0
        self._last_state: Optional[np.ndarray] = None
        self._initial_state: Optional[np.ndarray] = None  # D_0: 初始状态性能
        self._need_workload_restart = False  # 标志：是否需要重启工作负载

        # 历史状态跟踪（用于滑动窗口平均）
        self._throughput_history: List[float] = []  # 最近5步吞吐量
        self._latency_history: List[float] = []     # 最近5步延迟
        self._history_window = 5  # 滑动窗口大小

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
        
        # 在reset时，应用默认配置，确保初始状态基于默认参数
        # 这样第一步的action就是默认配置
        default_knobs = self.knob_space.get_default_knobs()
        if self._step_count == 0:  # 只在第一次reset时应用默认配置
            print("[MosquittoBrokerEnv] 应用默认Broker配置...")
            used_restart = apply_knobs(default_knobs)
            if used_restart:
                print("[MosquittoBrokerEnv] Broker已重启，等待稳定...")
                self._wait_for_broker_ready(max_wait_sec=self.cfg.broker_restart_stable_sec)
                # 等待$SYS主题发布
                print("[MosquittoBrokerEnv] 等待$SYS主题发布...")
                time.sleep(12.0)
            else:
                # 等待Broker稳定（如果只是reload）
                time.sleep(3.0)
        
        self._step_count = 0
        print("[MosquittoBrokerEnv] 开始采样初始状态...")
        state = self._sample_state()
        print("[MosquittoBrokerEnv] 初始状态采样完成")
        
        # 验证状态有效性
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"[MosquittoBrokerEnv] 警告: reset时检测到无效状态值（NaN/Inf），使用零状态")
            state = np.zeros(self.cfg.state_dim, dtype=np.float32)
        
        # 限制状态值范围
        state = np.clip(state, -1e6, 1e6)
        
        self._initial_state = state.copy()  # 保存初始状态 D_0
        self._last_state = state
        
        # gymnasium兼容：返回 (observation, info) 元组
        info: Dict[str, Any] = {}
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
        knobs = self.knob_space.decode_action(action)
        used_restart = apply_knobs(knobs)
        
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
            stable_wait_sec = self.cfg.broker_reload_stable_sec
            if self._step_count <= 3 or self._step_count % 10 == 0:  # 只在开始几步或每10步打印一次
                print(f"[MosquittoBrokerEnv] Broker 已重载配置，等待 {stable_wait_sec} 秒让系统稳定...")
            time.sleep(stable_wait_sec)

        # 3. 采样新状态（在等待工作负载稳定运行30秒之后进行）
        # 此时工作负载应该已经稳定运行，可以采集准确的指标
        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 开始采样新状态（步数: {self._step_count}）...")
        next_state = self._sample_state()

        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 新状态采样完成")

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
        reward = self._compute_reward(
            prev_state=self._last_state,
            next_state=next_state,
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
        info: Dict[str, Any] = {
            "knobs": knobs,
            "step": self._step_count,
        }

        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] step() 完成: reward={reward:.6f}, terminated={terminated}, truncated={truncated}")

        self._last_state = next_state
        return next_state, float(reward), bool(terminated), bool(truncated), info

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
                print("[MosquittoBrokerEnv] 建议: 检查Broker配置是否有 'sys_interval 10'")
        
        # 进程指标
        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 读取进程指标...")
        cpu_ratio, mem_ratio, ctxt_ratio = read_proc_metrics(
            self.cfg.proc
        )
        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 进程指标: CPU={cpu_ratio:.4f}, MEM={mem_ratio:.4f}, CTXT={ctxt_ratio:.4f}")
        # 拼接状态向量
        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[MosquittoBrokerEnv] 构建状态向量...")

        # 获取延迟和队列深度指标（需要扩展）
        # TODO: 这些指标需要通过工作负载管理器或扩展的采样机制获取
        latency_p50 = 10.0  # 默认值，单位：毫秒 (TODO: 实现实际测量)
        latency_p95 = 50.0  # 默认值，单位：毫秒 (TODO: 实现实际测量)
        queue_depth = 0.0   # 默认值 (TODO: 从 $SYS 主题获取)

        state = build_state_vector(
            broker_metrics,
            cpu_ratio,
            mem_ratio,
            ctxt_ratio,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            queue_depth=queue_depth,
            throughput_history=self._throughput_history,
            latency_history=self._latency_history,
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
        改进的奖励函数设计：
        使用绝对性能奖励 + 相对改进奖励 + 稳定性惩罚的组合

        奖励组成部分：
        1. 绝对性能奖励：基于当前吞吐量和延迟的绝对表现
        2. 相对改进奖励：相对于上一步的性能改进
        3. 稳定性惩罚：惩罚频繁的配置变化
        4. 资源约束惩罚：防止过度使用资源

        公式：
        reward = α * throughput_abs + β * (-latency_abs) +
                 γ * throughput_improvement + δ * (-latency_improvement) +
                 ε * stability_penalty + ζ * resource_penalty
        """
        # 1. 绝对性能奖励
        throughput_abs = self._extract_throughput(next_state)  # 吞吐量（归一化）
        latency_abs = self._extract_latency(next_state)       # 延迟（归一化）

        # 2. 相对改进奖励（如果有上一步状态）
        throughput_improvement = 0.0
        latency_improvement = 0.0

        if prev_state is not None:
            prev_throughput = self._extract_throughput(prev_state)
            prev_latency = self._extract_latency(prev_state)

            throughput_improvement = throughput_abs - prev_throughput
            latency_improvement = prev_latency - latency_abs  # 延迟降低是改进

        # 3. 稳定性惩罚（避免频繁配置变化）
        stability_penalty = 0.0
        if prev_state is not None and len(next_state) >= 11:  # 假设前11维是配置相关的状态
            # 计算配置变化程度（这里简化，使用吞吐量和延迟的变化作为代理）
            config_change = abs(throughput_improvement) + abs(latency_improvement)
            stability_penalty = -2.0 * config_change  # 惩罚大的变化

        # 4. 资源约束惩罚
        cpu_ratio = float(next_state[2])
        mem_ratio = float(next_state[3])

        resource_penalty = 0.0
        # CPU 约束：超过 90% 时惩罚
        if cpu_ratio > 0.9:
            resource_penalty -= 50.0 * (cpu_ratio - 0.9)
        # 内存约束：超过 90% 时惩罚
        if mem_ratio > 0.9:
            resource_penalty -= 50.0 * (mem_ratio - 0.9)

        # 5. 权重系数（调优后的值）
        alpha = 100.0   # 绝对吞吐量权重
        beta = 50.0     # 绝对延迟权重（负值）
        gamma = 50.0    # 吞吐量改进权重
        delta = 30.0    # 延迟改进权重
        epsilon = 1.0   # 稳定性惩罚权重
        zeta = 1.0      # 资源惩罚权重

        # 6. 计算最终奖励
        performance_reward = (
            alpha * throughput_abs +                    # 绝对吞吐量奖励
            beta * (-latency_abs) +                     # 绝对延迟惩罚（延迟越低越好）
            gamma * throughput_improvement +            # 吞吐量改进奖励
            delta * latency_improvement +               # 延迟改进奖励
            epsilon * stability_penalty                 # 稳定性惩罚
        )

        reward = performance_reward + zeta * resource_penalty

        # 7. 确保奖励是有效数值
        reward = float(reward)
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        if self._step_count <= 3 or self._step_count % 20 == 0:
            print(f"[Reward] 吞吐量: {throughput_abs:.6f}, 延迟: {latency_abs:.6f}, "
                  f"改进: 吞吐量{throughput_improvement:+.6f}, 延迟{latency_improvement:+.6f}, "
                  f"稳定性惩罚: {stability_penalty:.6f}, 资源惩罚: {resource_penalty:.6f}, "
                  f"总奖励: {reward:.6f}")

        return reward
    
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

