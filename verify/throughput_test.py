#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
吞吐量测试脚本

测试在不同Broker配置和工作负载组合下的吞吐量性能。

测试配置：
1. Broker配置1：max_inflight_messages=100，其他参数默认
2. Broker配置2：所有参数默认

工作负载组合（每种配置测试12种）：
- 消息大小：256B, 512B, 1024B
- QoS：0, 1
- 发布周期：10ms, 50ms
- 发布端：100个
- 接收端：10个

每个测试：
1. 应用Broker配置
2. 启动工作负载
3. 稳定运行30秒
4. 统计吞吐量
5. 停止工作负载
6. 记录结果到CSV
"""

from __future__ import annotations

import sys
import os
import time
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# 尝试导入tqdm用于进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("提示: tqdm未安装，将使用文本进度显示。安装命令: pip install tqdm")

# 添加项目根目录到路径（必须在所有导入之前）
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)

# 确保项目根目录在sys.path中（使用绝对路径）
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# 如果PYTHONPATH环境变量存在，也添加到sys.path
pythonpath = os.environ.get('PYTHONPATH', '')
if pythonpath:
    for path in pythonpath.split(os.pathsep):
        if path and path not in sys.path:
            sys.path.insert(0, path)

# 添加用户的site-packages路径（即使使用sudo也能访问用户安装的包）
# 这很重要，因为numpy等包可能安装在用户的.local目录中
# bash脚本会传递HOME环境变量（原始用户的主目录）
user_home = os.environ.get('HOME', '')
if not user_home or user_home == '/root':
    # 如果HOME是/root或未设置，尝试从SUDO_USER获取
    sudo_user = os.environ.get('SUDO_USER', '')
    if sudo_user:
        try:
            import pwd
            user_home = pwd.getpwnam(sudo_user).pw_dir
        except (KeyError, ImportError):
            # 如果无法获取，尝试常见的用户主目录路径
            user_home = f"/home/{sudo_user}"

if user_home:
    user_site_packages = Path(user_home) / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if user_site_packages.exists() and str(user_site_packages) not in sys.path:
        sys.path.insert(0, str(user_site_packages))

# 直接导入需要的模块，避免触发 environment/__init__.py（它会导入broker.py，需要gym）
# 这样可以避免安装gym依赖（throughput_test.py不需要gym）
try:
    import importlib.util
    import types
    
    # 创建environment包的占位符，避免__init__.py被导入
    if 'environment' not in sys.modules:
        env_pkg = types.ModuleType('environment')
        env_pkg.__path__ = [str(project_root / "environment")]
        sys.modules['environment'] = env_pkg
    
    # 先导入config（不依赖gym）
    config_path = project_root / "environment" / "config.py"
    spec = importlib.util.spec_from_file_location("environment.config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules['environment.config'] = config_module
    spec.loader.exec_module(config_module)
    MQTTConfig = config_module.MQTTConfig
    
    # 然后导入utils（依赖config，但不依赖gym）
    utils_path = project_root / "environment" / "utils.py"
    spec = importlib.util.spec_from_file_location("environment.utils", utils_path)
    utils_module = importlib.util.module_from_spec(spec)
    sys.modules['environment.utils'] = utils_module
    spec.loader.exec_module(utils_module)
    MQTTSampler = utils_module.MQTTSampler
    
    # 最后导入knobs（不依赖gym）
    knobs_path = project_root / "environment" / "knobs.py"
    spec = importlib.util.spec_from_file_location("environment.knobs", knobs_path)
    knobs_module = importlib.util.module_from_spec(spec)
    sys.modules['environment.knobs'] = knobs_module
    spec.loader.exec_module(knobs_module)
    apply_knobs = knobs_module.apply_knobs
    BrokerKnobSpace = knobs_module.BrokerKnobSpace
    
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本文件路径: {__file__}")
    print(f"项目根目录: {project_root_str}")
    print(f"sys.path: {sys.path[:5]}...")
    print(f"PYTHONPATH环境变量: {os.environ.get('PYTHONPATH', '未设置')}")
    # 检查environment目录是否存在
    env_dir = project_root / "environment"
    print(f"environment目录是否存在: {env_dir.exists()}")
    if env_dir.exists():
        print(f"environment目录内容: {list(env_dir.iterdir())}")
    import traceback
    traceback.print_exc()
    raise
from script.workload import WorkloadManager, WorkloadConfig


@dataclass
class TestCase:
    """测试用例"""
    message_size: int  # 字节
    qos: int  # 0, 1, 或 2
    publisher_interval_ms: int  # 毫秒
    num_publishers: int = 100
    num_subscribers: int = 10


@dataclass
class BrokerConfig:
    """Broker配置"""
    name: str
    max_inflight_messages: int | None = None  # None表示使用默认值


class SubscriberMessageCounter:
    """统计订阅者接收到的消息总数"""
    
    def __init__(self, broker_host: str = "127.0.0.1", broker_port: int = 1883):
        """
        初始化消息计数器
        
        Args:
            broker_host: MQTT Broker 地址
            broker_port: MQTT Broker 端口
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self._message_count = 0
        self._client = None
        self._connected = False
        
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT连接回调"""
        if rc == 0:
            self._connected = True
        else:
            print(f"[SubscriberMessageCounter] 连接失败: rc={rc}")
    
    def _on_message(self, client, userdata, msg):
        """MQTT消息回调"""
        self._message_count += 1
    
    def count_messages(self, topic: str, duration_sec: float) -> int:
        """
        在指定时间内统计接收到的消息总数
        
        Args:
            topic: 要订阅的主题
            duration_sec: 统计持续时间（秒）
            
        Returns:
            接收到的消息总数
        """
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            print("[SubscriberMessageCounter] 错误: paho-mqtt未安装，无法统计订阅者消息")
            return 0
        
        self._message_count = 0
        self._connected = False
        
        # 创建MQTT客户端
        client_id = f"throughput_counter_{int(time.time())}"
        self._client = mqtt.Client(client_id=client_id)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        
        try:
            # 连接到Broker
            self._client.connect(self.broker_host, self.broker_port, keepalive=60)
            self._client.loop_start()
            
            # 等待连接建立
            connect_timeout = 5.0
            start_time = time.time()
            while not self._connected and (time.time() - start_time) < connect_timeout:
                time.sleep(0.1)
            
            if not self._connected:
                print(f"[SubscriberMessageCounter] 警告: 连接超时")
                return 0
            
            # 订阅主题
            self._client.subscribe(topic, qos=0)
            
            # 等待指定时间，统计消息
            time.sleep(duration_sec)
            
            # 停止并断开连接
            self._client.loop_stop()
            self._client.disconnect()
            
            return self._message_count
            
        except Exception as e:
            print(f"[SubscriberMessageCounter] 错误: {e}")
            if self._client:
                try:
                    self._client.loop_stop()
                    self._client.disconnect()
                except:
                    pass
            return 0


class ThroughputTester:
    """吞吐量测试器"""
    
    def __init__(self, output_csv: str = "throughput_test_results.csv"):
        """
        初始化测试器
        
        Args:
            output_csv: 输出CSV文件路径（相对于verify目录）
        """
        # 确保输出文件在verify目录下
        if not Path(output_csv).is_absolute():
            output_csv = Path(__file__).parent / output_csv
        self.output_csv = Path(output_csv)
        self.results: List[Dict[str, Any]] = []
        self._last_broker_config: str | None = None  # 记录上一次的Broker配置名称
        
        # 初始化MQTT配置
        self.mqtt_config = MQTTConfig(
            host="127.0.0.1",
            port=1883,
            topics=[
                "$SYS/broker/messages/received",
                "$SYS/broker/messages/sent",
                "$SYS/broker/messages/publish/received",
                "$SYS/broker/messages/publish/sent",
            ],
            timeout_sec=5.0,
        )
        
        # 初始化工作负载管理器
        self.workload_manager = WorkloadManager(
            broker_host="127.0.0.1",
            broker_port=1883,
        )
        
        # 初始化knob space（用于获取默认配置）
        self.knob_space = BrokerKnobSpace()
    
    def apply_broker_config(self, config: BrokerConfig, force_restart: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """
        应用Broker配置
        
        Args:
            config: Broker配置
            force_restart: 是否强制重启Broker（用于配置切换时）
            
        Returns:
            (是否使用了重启, 实际应用的配置项字典)
        """
        print(f"\n{'='*80}")
        print(f"应用Broker配置: {config.name}")
        print(f"{'='*80}")
        
        import subprocess
        config_path = Path("/etc/mosquitto/conf.d/broker_tuner.conf")
        
        # 检查是否需要切换配置（从一种配置切换到另一种）
        config_changed = (self._last_broker_config is not None and 
                         self._last_broker_config != config.name)
        
        if config_changed:
            print(f"  ⚠️  检测到配置切换: {self._last_broker_config} -> {config.name}")
            print(f"  🔄 将强制重启Broker以确保配置完全生效")
            force_restart = True
        
        # 获取默认配置值
        default_knobs = self.knob_space.get_default_knobs()
        
        if config.max_inflight_messages is not None:
            # 只设置max_inflight_messages，其他使用默认值
            knobs = {
                "max_inflight_messages": config.max_inflight_messages,
            }
            print(f"  设置 max_inflight_messages = {config.max_inflight_messages}")
            
            # 如果需要强制重启，使用force_restart参数
            if force_restart:
                print(f"  强制重启Broker...")
                used_restart = apply_knobs(knobs, force_restart=True)
            else:
                used_restart = apply_knobs(knobs)
            
            # 构建完整的配置字典（包含所有配置项）
            applied_knobs = default_knobs.copy()
            applied_knobs["max_inflight_messages"] = config.max_inflight_messages
        else:
            # 使用默认配置（清空配置文件，让Mosquitto使用系统默认值）
            print(f"  使用默认配置（清空自定义配置，使用系统默认值）")
            applied_knobs = default_knobs.copy()
            
            # 清空配置文件
            if config_path.exists():
                try:
                    # 备份原配置
                    backup_path = config_path.with_suffix(".conf.backup")
                    subprocess.run(
                        ["sudo", "cp", str(config_path), str(backup_path)],
                        check=True,
                        capture_output=True
                    )
                    
                    # 清空配置文件（只保留注释）
                    subprocess.run(
                        ["sudo", "bash", "-c", f"echo '# 默认配置（所有参数使用系统默认值）' > {config_path}"],
                        check=True,
                        capture_output=True
                    )
                    print(f"  配置文件已清空")
                except Exception as e:
                    print(f"  ⚠️  清空配置文件失败: {e}")
            
            # 如果需要强制重启，直接重启；否则尝试reload
            if force_restart:
                print(f"  强制重启Broker...")
                try:
                    subprocess.run(
                        ["sudo", "systemctl", "restart", "mosquitto"],
                        check=True,
                        capture_output=True
                    )
                    used_restart = True
                except Exception as e:
                    print(f"  ❌ 重启Broker失败: {e}")
                    raise
            else:
                # 重载配置（不需要重启，因为只是清空了配置）
                try:
                    subprocess.run(
                        ["sudo", "systemctl", "reload", "mosquitto"],
                        check=True,
                        capture_output=True
                    )
                    used_restart = False
                except Exception as e:
                    print(f"  ⚠️  重载配置失败，尝试重启: {e}")
                    subprocess.run(
                        ["sudo", "systemctl", "restart", "mosquitto"],
                        check=True,
                        capture_output=True
                    )
                    used_restart = True
        
        if used_restart:
            print(f"  Broker已重启，等待稳定...")
            time.sleep(5.0)  # 等待Broker重启稳定
            
            # 验证Broker是否正常运行
            import subprocess as sp
            max_wait = 20
            waited = 0
            while waited < max_wait:
                try:
                    result = sp.run(
                        ["systemctl", "is-active", "mosquitto"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0 and result.stdout.strip() == "active":
                        # 检查端口是否监听
                        port_check = sp.run(
                            ["sudo", "netstat", "-tlnp"],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if "1883" in port_check.stdout:
                            print(f"  ✅ Broker已正常运行（端口1883已监听）")
                            break
                except:
                    pass
                time.sleep(1.0)
                waited += 1
                if waited % 5 == 0:
                    print(f"  等待Broker就绪... ({waited}/{max_wait}秒)")
            else:
                print(f"  ⚠️  Broker可能未完全就绪，继续执行...")
            
            # 等待$SYS主题发布
            print(f"  等待$SYS主题发布...")
            time.sleep(12.0)
        else:
            print(f"  Broker配置已重载，等待稳定...")
            time.sleep(3.0)
        
        # 更新最后使用的配置名称
        self._last_broker_config = config.name
        
        print(f"  ✅ Broker配置应用完成")
        return used_restart, applied_knobs
    
    def run_test_case(
        self,
        broker_config: BrokerConfig,
        test_case: TestCase,
        stable_time_sec: float = 30.0
    ) -> Dict[str, Any]:
        """
        运行单个测试用例
        
        Args:
            broker_config: Broker配置
            test_case: 测试用例
            stable_time_sec: 稳定运行时间（秒）
            
        Returns:
            测试结果字典
        """
        print(f"\n{'='*80}")
        print(f"测试用例: {test_case.message_size}B, QoS={test_case.qos}, 周期={test_case.publisher_interval_ms}ms")
        print(f"{'='*80}")
        
        # 1. 应用Broker配置（如果切换配置，会强制重启Broker）
        broker_restarted, applied_knobs = self.apply_broker_config(broker_config)
        
        # 2. 确保之前的工作负载已完全停止（重启工作负载）
        print(f"\n确保工作负载已完全停止...")
        try:
            if self.workload_manager.is_running():
                print(f"  检测到正在运行的工作负载，正在停止...")
                self.workload_manager.stop()
                print(f"  ✅ 旧工作负载已停止")
            else:
                print(f"  ✅ 没有正在运行的工作负载")
        except Exception as e:
            print(f"  ⚠️  停止旧工作负载时出错（可能已经停止）: {e}")
        
        # 如果Broker重启了，需要额外等待，确保Broker完全就绪
        if broker_restarted:
            print(f"  Broker已重启，额外等待确保Broker完全就绪（5秒）...")
            time.sleep(5.0)
        
        # 等待一段时间，确保进程完全终止
        print(f"  等待进程完全终止（3秒）...")
        time.sleep(3.0)
        
        # 3. 创建工作负载配置
        workload_config = WorkloadConfig(
            num_publishers=test_case.num_publishers,
            num_subscribers=test_case.num_subscribers,
            topic="test/throughput",
            message_size=test_case.message_size,
            qos=test_case.qos,
            publisher_interval_ms=test_case.publisher_interval_ms,
            duration=0,  # 持续运行直到手动停止
        )
        
        # 4. 启动新的工作负载
        print(f"\n启动新的工作负载...")
        print(f"  发布者: {test_case.num_publishers}")
        print(f"  订阅者: {test_case.num_subscribers}")
        print(f"  消息大小: {test_case.message_size}B")
        print(f"  QoS: {test_case.qos}")
        print(f"  发布周期: {test_case.publisher_interval_ms}ms")
        
        try:
            self.workload_manager.start(config=workload_config)
            print(f"  ✅ 工作负载启动成功")
        except Exception as e:
            print(f"  ❌ 工作负载启动失败: {e}")
            # 获取当前应用的配置
            default_knobs = self.knob_space.get_default_knobs()
            if broker_config.max_inflight_messages is not None:
                default_knobs["max_inflight_messages"] = broker_config.max_inflight_messages
            
            return {
                "broker_config": broker_config.name,
                "message_size": test_case.message_size,
                "qos": test_case.qos,
                "publisher_interval_ms": test_case.publisher_interval_ms,
                "num_publishers": test_case.num_publishers,
                "num_subscribers": test_case.num_subscribers,
                "throughput": 0.0,
                "error": str(e),
                # 所有Broker配置项
                "max_inflight_messages": default_knobs.get("max_inflight_messages", 0),
                "max_inflight_bytes": default_knobs.get("max_inflight_bytes", 0),
                "max_queued_messages": default_knobs.get("max_queued_messages", 0),
                "max_queued_bytes": default_knobs.get("max_queued_bytes", 0),
                "queue_qos0_messages": default_knobs.get("queue_qos0_messages", False),
                "memory_limit": default_knobs.get("memory_limit", 0),
                "persistence": default_knobs.get("persistence", False),
                "autosave_interval": default_knobs.get("autosave_interval", 0),
                "set_tcp_nodelay": default_knobs.get("set_tcp_nodelay", False),
                "max_packet_size": default_knobs.get("max_packet_size", 0),
                "message_size_limit": default_knobs.get("message_size_limit", 0),
            }
        
        # 5. 等待工作负载稳定
        print(f"\n等待工作负载稳定运行 {stable_time_sec} 秒...")
        time.sleep(stable_time_sec)
        
        # 6. 统计吞吐量（使用订阅者接收到的消息总数）
        print(f"\n开始统计吞吐量（订阅者接收消息数）...")
        throughput = 0.0
        sample_duration = 12.0  # 采样持续时间（秒）
        
        try:
            # 使用SubscriberMessageCounter统计订阅者接收到的消息总数
            counter = SubscriberMessageCounter(
                broker_host=self.workload_manager.broker_host,
                broker_port=self.workload_manager.broker_port
            )
            
            # 获取测试主题（与工作负载使用的主题相同）
            # 使用工作负载配置中的主题，默认为 "test/throughput"
            test_topic = "test/throughput"  # 与WorkloadConfig中使用的主题一致
            
            print(f"  订阅主题: {test_topic}")
            print(f"  统计时间: {sample_duration}秒")
            print(f"  开始统计订阅者接收到的消息总数...")
            
            # 统计在指定时间内单个订阅者接收到的消息数
            messages_per_subscriber = counter.count_messages(test_topic, sample_duration)
            
            # 计算所有订阅者接收到的消息总数
            # 在MQTT中，Broker会将每条消息发送给所有订阅者
            # 所以：所有订阅者收到的消息总数 = 单个订阅者收到的消息数 × 订阅者数量
            num_subscribers = test_case.num_subscribers
            total_messages = messages_per_subscriber * num_subscribers
            
            # 计算吞吐量（每秒所有订阅者接收到的消息总数）
            throughput = total_messages / sample_duration
            
            print(f"  单个订阅者接收到的消息数: {messages_per_subscriber} 条")
            print(f"  订阅者数量: {num_subscribers}")
            print(f"  所有订阅者接收到的消息总数: {total_messages} 条")
            print(f"  统计时长: {sample_duration} 秒")
            print(f"  ✅ 吞吐量统计完成: {throughput:.2f} msg/s (所有订阅者的总和)")
            
            if total_messages == 0:
                print(f"  ⚠️  警告: 在 {sample_duration} 秒内未收到任何消息")
                print(f"  可能原因:")
                print(f"    1. 工作负载未正常运行")
                print(f"    2. 主题不匹配（工作负载主题: {test_topic}）")
                print(f"    3. 消息发布频率太低")
            
        except Exception as e:
            print(f"  ❌ 吞吐量统计失败: {e}")
            import traceback
            traceback.print_exc()
            throughput = 0.0
        
        # 7. 停止工作负载
        print(f"\n停止工作负载...")
        try:
            self.workload_manager.stop()
            print(f"  ✅ 工作负载已停止")
        except Exception as e:
            print(f"  ⚠️  停止工作负载时出错: {e}")
        
        # 8. 等待一段时间，确保进程完全终止和Broker稳定
        print(f"  等待进程完全终止和Broker稳定（3秒）...")
        time.sleep(3.0)
        
        # 9. 返回结果（包含所有配置项）
        result = {
            "broker_config": broker_config.name,
            "message_size": test_case.message_size,
            "qos": test_case.qos,
            "publisher_interval_ms": test_case.publisher_interval_ms,
            "num_publishers": test_case.num_publishers,
            "num_subscribers": test_case.num_subscribers,
            "throughput": throughput,
            # 所有Broker配置项
            "max_inflight_messages": applied_knobs.get("max_inflight_messages", 0),
            "max_inflight_bytes": applied_knobs.get("max_inflight_bytes", 0),
            "max_queued_messages": applied_knobs.get("max_queued_messages", 0),
            "max_queued_bytes": applied_knobs.get("max_queued_bytes", 0),
            "queue_qos0_messages": applied_knobs.get("queue_qos0_messages", False),
            "memory_limit": applied_knobs.get("memory_limit", 0),
            "persistence": applied_knobs.get("persistence", False),
            "autosave_interval": applied_knobs.get("autosave_interval", 0),
            "set_tcp_nodelay": applied_knobs.get("set_tcp_nodelay", False),
            "max_packet_size": applied_knobs.get("max_packet_size", 0),
            "message_size_limit": applied_knobs.get("message_size_limit", 0),
        }
        
        return result
    
    def run_all_tests(self):
        """运行所有测试"""
        print(f"\n{'='*80}")
        print(f"开始吞吐量测试")
        print(f"{'='*80}")
        
        # 定义Broker配置
        broker_configs = [
            BrokerConfig(
                name="max_inflight_100",
                max_inflight_messages=100,
            ),
            BrokerConfig(
                name="default",
                max_inflight_messages=None,  # 使用默认值
            ),
        ]
        
        # 定义测试用例
        test_cases = [
            TestCase(message_size=256, qos=0, publisher_interval_ms=10),
            TestCase(message_size=256, qos=0, publisher_interval_ms=50),
            TestCase(message_size=256, qos=1, publisher_interval_ms=10),
            TestCase(message_size=256, qos=1, publisher_interval_ms=50),
            TestCase(message_size=512, qos=0, publisher_interval_ms=10),
            TestCase(message_size=512, qos=0, publisher_interval_ms=50),
            TestCase(message_size=512, qos=1, publisher_interval_ms=10),
            TestCase(message_size=512, qos=1, publisher_interval_ms=50),
            TestCase(message_size=1024, qos=0, publisher_interval_ms=10),
            TestCase(message_size=1024, qos=0, publisher_interval_ms=50),
            TestCase(message_size=1024, qos=1, publisher_interval_ms=10),
            TestCase(message_size=1024, qos=1, publisher_interval_ms=50),
        ]
        
        total_tests = len(broker_configs) * len(test_cases)
        current_test = 0
        
        # 创建进度条
        if TQDM_AVAILABLE:
            pbar = tqdm(
                total=total_tests,
                desc="测试进度",
                unit="测试",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ncols=100
            )
        else:
            pbar = None
            print(f"\n开始测试，共 {total_tests} 个测试用例\n")
        
        # 运行所有测试
        for broker_config_idx, broker_config in enumerate(broker_configs):
            # 如果是第一个配置，或者切换配置，需要确保工作负载已停止
            if broker_config_idx > 0:
                print(f"\n{'='*80}")
                print(f"切换到新的Broker配置: {broker_config.name}")
                print(f"{'='*80}")
                print(f"⚠️  配置切换将强制重启Broker，并重启工作负载")
                
                # 确保之前的工作负载已停止
                try:
                    if self.workload_manager.is_running():
                        print(f"停止之前的工作负载...")
                        self.workload_manager.stop()
                        print(f"✅ 工作负载已停止")
                except Exception as e:
                    print(f"⚠️  停止工作负载时出错: {e}")
                
                # 等待进程完全终止
                print(f"等待进程完全终止（5秒）...")
                time.sleep(5.0)
            
            for test_case in test_cases:
                current_test += 1
                
                # 更新进度条描述
                if pbar is not None:
                    test_desc = f"{broker_config.name} | {test_case.message_size}B QoS{test_case.qos} {test_case.publisher_interval_ms}ms"
                    pbar.set_description(f"测试进度 [{test_desc}]")
                else:
                    print(f"\n\n{'#'*80}")
                    print(f"测试进度: {current_test}/{total_tests}")
                    print(f"配置: {broker_config.name} | {test_case.message_size}B, QoS={test_case.qos}, 周期={test_case.publisher_interval_ms}ms")
                    print(f"{'#'*80}")
                
                try:
                    # 每个测试用例都会重启工作负载（在run_test_case内部处理）
                    result = self.run_test_case(broker_config, test_case)
                    self.results.append(result)
                    
                    # 保存中间结果（每完成一个测试就保存）
                    self.save_results()
                    
                    # 更新进度条
                    if pbar is not None:
                        pbar.update(1)
                        # 显示当前吞吐量
                        throughput = result.get("throughput", 0.0)
                        pbar.set_postfix({"吞吐量": f"{throughput:.2f} msg/s"})
                    
                except Exception as e:
                    print(f"\n❌ 测试失败: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 记录错误结果（包含默认配置项）
                    default_knobs = self.knob_space.get_default_knobs()
                    if broker_config.max_inflight_messages is not None:
                        default_knobs["max_inflight_messages"] = broker_config.max_inflight_messages
                    
                    error_result = {
                        "broker_config": broker_config.name,
                        "message_size": test_case.message_size,
                        "qos": test_case.qos,
                        "publisher_interval_ms": test_case.publisher_interval_ms,
                        "num_publishers": test_case.num_publishers,
                        "num_subscribers": test_case.num_subscribers,
                        "throughput": 0.0,
                        "error": str(e),
                        # 所有Broker配置项
                        "max_inflight_messages": default_knobs.get("max_inflight_messages", 0),
                        "max_inflight_bytes": default_knobs.get("max_inflight_bytes", 0),
                        "max_queued_messages": default_knobs.get("max_queued_messages", 0),
                        "max_queued_bytes": default_knobs.get("max_queued_bytes", 0),
                        "queue_qos0_messages": default_knobs.get("queue_qos0_messages", False),
                        "memory_limit": default_knobs.get("memory_limit", 0),
                        "persistence": default_knobs.get("persistence", False),
                        "autosave_interval": default_knobs.get("autosave_interval", 0),
                        "set_tcp_nodelay": default_knobs.get("set_tcp_nodelay", False),
                        "max_packet_size": default_knobs.get("max_packet_size", 0),
                        "message_size_limit": default_knobs.get("message_size_limit", 0),
                    }
                    self.results.append(error_result)
                    self.save_results()
                    
                    # 更新进度条（即使失败也更新）
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({"状态": "失败"})
                
                # 每个测试用例之间额外等待，确保工作负载完全清理
                if current_test < total_tests:
                    if pbar is None:  # 只在没有进度条时打印
                        print(f"\n等待工作负载完全清理（2秒）...")
                    time.sleep(2.0)
        
        # 关闭进度条
        if pbar is not None:
            pbar.close()
            print()  # 换行
        
        # 最终清理：确保所有工作负载已停止
        print(f"\n\n最终清理：确保所有工作负载已停止...")
        try:
            if self.workload_manager.is_running():
                self.workload_manager.stop()
                print(f"  ✅ 所有工作负载已停止")
            else:
                print(f"  ✅ 没有正在运行的工作负载")
        except Exception as e:
            print(f"  ⚠️  清理工作负载时出错: {e}")
        
        print(f"\n\n{'='*80}")
        print(f"所有测试完成！")
        print(f"{'='*80}")
        print(f"结果已保存到: {self.output_csv}")
        self.print_summary()
    
    def save_results(self):
        """保存结果到CSV文件"""
        if not self.results:
            return
        
        # 确保目录存在
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入CSV（包含所有配置项）
        fieldnames = [
            "broker_config",
            "message_size",
            "qos",
            "publisher_interval_ms",
            "num_publishers",
            "num_subscribers",
            "throughput",
            "error",
            # 所有Broker配置项
            "max_inflight_messages",
            "max_inflight_bytes",
            "max_queued_messages",
            "max_queued_bytes",
            "queue_qos0_messages",
            "memory_limit",
            "persistence",
            "autosave_interval",
            "set_tcp_nodelay",
            "max_packet_size",
            "message_size_limit",
        ]
        
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)
        
        print(f"\n✅ 结果已保存到: {self.output_csv}")
    
    def print_summary(self):
        """打印测试摘要"""
        if not self.results:
            print("没有测试结果")
            return
        
        print(f"\n测试摘要:")
        print(f"{'='*80}")
        print(f"{'配置':<20} {'消息大小':<10} {'QoS':<5} {'周期(ms)':<10} {'吞吐量(msg/s)':<15} {'max_inflight':<12}")
        print(f"{'-'*80}")
        
        for result in self.results:
            config = result.get("broker_config", "unknown")
            msg_size = result.get("message_size", 0)
            qos = result.get("qos", 0)
            interval = result.get("publisher_interval_ms", 0)
            throughput = result.get("throughput", 0.0)
            max_inflight = result.get("max_inflight_messages", 0)
            
            print(f"{config:<20} {msg_size:<10} {qos:<5} {interval:<10} {throughput:<15.2f} {max_inflight:<12}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="吞吐量测试")
    parser.add_argument(
        "--output",
        type=str,
        default="throughput_test_results.csv",
        help="输出CSV文件路径（默认：throughput_test_results.csv）",
    )
    parser.add_argument(
        "--stable-time",
        type=float,
        default=30.0,
        help="工作负载稳定运行时间（秒，默认：30.0）",
    )
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ThroughputTester(output_csv=args.output)
    
    # 运行所有测试
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        print("保存已完成的测试结果...")
        tester.save_results()
        tester.print_summary()
    except Exception as e:
        print(f"\n\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        print("保存已完成的测试结果...")
        tester.save_results()
        tester.print_summary()
        sys.exit(1)


if __name__ == "__main__":
    main()
