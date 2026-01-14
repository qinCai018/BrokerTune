#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的吞吐量测试脚本
只测试两种配置和一种工作负载组合
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# 添加用户的site-packages路径
user_site_packages = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if user_site_packages.exists() and str(user_site_packages) not in sys.path:
    sys.path.insert(0, str(user_site_packages))

# 导入必要的模块
try:
    import importlib.util
    import types
    
    # 创建environment包的占位符
    if 'environment' not in sys.modules:
        env_pkg = types.ModuleType('environment')
        env_pkg.__path__ = [str(project_root / "environment")]
        sys.modules['environment'] = env_pkg
    
    # 导入config
    config_path = project_root / "environment" / "config.py"
    spec = importlib.util.spec_from_file_location("environment.config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules['environment.config'] = config_module
    spec.loader.exec_module(config_module)
    
    # 导入utils
    utils_path = project_root / "environment" / "utils.py"
    spec = importlib.util.spec_from_file_location("environment.utils", utils_path)
    utils_module = importlib.util.module_from_spec(spec)
    sys.modules['environment.utils'] = utils_module
    spec.loader.exec_module(utils_module)
    
    # 导入knobs
    knobs_path = project_root / "environment" / "knobs.py"
    spec = importlib.util.spec_from_file_location("environment.knobs", knobs_path)
    knobs_module = importlib.util.module_from_spec(spec)
    sys.modules['environment.knobs'] = knobs_module
    spec.loader.exec_module(knobs_module)
    apply_knobs = knobs_module.apply_knobs
    BrokerKnobSpace = knobs_module.BrokerKnobSpace
    
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

from script.workload import WorkloadManager, WorkloadConfig


class SubscriberMessageCounter:
    """统计订阅者接收到的消息总数"""
    
    def __init__(self, broker_host: str = "127.0.0.1", broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self._message_count = 0
        self._client = None
        self._connected = False
        
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
        else:
            print(f"[SubscriberMessageCounter] 连接失败: rc={rc}")
    
    def _on_message(self, client, userdata, msg):
        self._message_count += 1
    
    def count_messages(self, topic: str, duration_sec: float) -> int:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            print("[SubscriberMessageCounter] 错误: paho-mqtt未安装")
            return 0
        
        self._message_count = 0
        self._connected = False
        
        client_id = f"throughput_counter_{int(time.time())}"
        self._client = mqtt.Client(client_id=client_id)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        
        try:
            self._client.connect(self.broker_host, self.broker_port, keepalive=60)
            self._client.loop_start()
            
            connect_timeout = 5.0
            start_time = time.time()
            while not self._connected and (time.time() - start_time) < connect_timeout:
                time.sleep(0.1)
            
            if not self._connected:
                print(f"[SubscriberMessageCounter] 警告: 连接超时")
                return 0
            
            self._client.subscribe(topic, qos=0)
            time.sleep(duration_sec)
            
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


def test_configuration(config_name: str, max_inflight_messages: int | None, 
                      message_size: int, qos: int, publisher_interval_ms: int):
    """测试单个配置"""
    print(f"\n{'='*80}")
    print(f"测试配置: {config_name}")
    print(f"{'='*80}")
    
    # 获取默认配置
    knob_space = BrokerKnobSpace()
    default_knobs = knob_space.get_default_knobs()
    
    # 应用配置
    if max_inflight_messages is not None:
        print(f"\n应用配置: max_inflight_messages = {max_inflight_messages}")
        knobs = {"max_inflight_messages": max_inflight_messages}
        apply_knobs(knobs, force_restart=True)
        applied_knobs = default_knobs.copy()
        applied_knobs["max_inflight_messages"] = max_inflight_messages
    else:
        print(f"\n使用默认配置（清空自定义配置）")
        # 清空配置文件
        import subprocess
        config_path = Path("/etc/mosquitto/conf.d/broker_tuner.conf")
        if config_path.exists():
            subprocess.run(
                ["sudo", "bash", "-c", f"echo '# 默认配置' > {config_path}"],
                check=True,
                capture_output=True
            )
        subprocess.run(
            ["sudo", "systemctl", "restart", "mosquitto"],
            check=True,
            capture_output=True
        )
        applied_knobs = default_knobs.copy()
    
    # 等待Broker稳定
    print("等待Broker稳定（5秒）...")
    time.sleep(5.0)
    
    # 创建工作负载管理器
    workload_manager = WorkloadManager(
        broker_host="127.0.0.1",
        broker_port=1883,
    )
    
    # 创建工作负载配置
    workload_config = WorkloadConfig(
        num_publishers=100,
        num_subscribers=10,
        topic="test/throughput",
        message_size=message_size,
        qos=qos,
        publisher_interval_ms=publisher_interval_ms,
        duration=0,
    )
    
    # 启动工作负载
    print(f"\n启动工作负载:")
    print(f"  消息大小: {message_size}B")
    print(f"  QoS: {qos}")
    print(f"  发布周期: {publisher_interval_ms}ms")
    print(f"  发布者: 100, 订阅者: 10")
    
    try:
        workload_manager.start(config=workload_config)
        print("  ✅ 工作负载启动成功")
    except Exception as e:
        print(f"  ❌ 工作负载启动失败: {e}")
        return None
    
    # 等待工作负载稳定
    print("\n等待工作负载稳定运行（30秒）...")
    time.sleep(30.0)
    
    # 统计吞吐量
    print("\n开始统计吞吐量...")
    sample_duration = 12.0
    
    try:
        counter = SubscriberMessageCounter(
            broker_host="127.0.0.1",
            broker_port=1883
        )
        
        messages_per_subscriber = counter.count_messages("test/throughput", sample_duration)
        total_messages = messages_per_subscriber * 10  # 10个订阅者
        throughput = total_messages / sample_duration
        
        print(f"  单个订阅者接收到的消息数: {messages_per_subscriber} 条")
        print(f"  订阅者数量: 10")
        print(f"  所有订阅者接收到的消息总数: {total_messages} 条")
        print(f"  统计时长: {sample_duration} 秒")
        print(f"  ✅ 吞吐量: {throughput:.2f} msg/s")
        
    except Exception as e:
        print(f"  ❌ 吞吐量统计失败: {e}")
        import traceback
        traceback.print_exc()
        throughput = 0.0
    
    # 停止工作负载
    print("\n停止工作负载...")
    try:
        workload_manager.stop()
        print("  ✅ 工作负载已停止")
    except Exception as e:
        print(f"  ⚠️  停止工作负载时出错: {e}")
    
    # 打印配置信息
    print(f"\n配置信息:")
    print(f"  max_inflight_messages: {applied_knobs['max_inflight_messages']}")
    print(f"  max_inflight_bytes: {applied_knobs['max_inflight_bytes']}")
    print(f"  max_queued_messages: {applied_knobs['max_queued_messages']}")
    print(f"  max_queued_bytes: {applied_knobs['max_queued_bytes']}")
    print(f"  queue_qos0_messages: {applied_knobs['queue_qos0_messages']}")
    print(f"  memory_limit: {applied_knobs['memory_limit']} (0=无限制)")
    print(f"  persistence: {applied_knobs['persistence']}")
    print(f"  autosave_interval: {applied_knobs['autosave_interval']}")
    print(f"  set_tcp_nodelay: {applied_knobs['set_tcp_nodelay']}")
    print(f"  max_packet_size: {applied_knobs['max_packet_size']}")
    print(f"  message_size_limit: {applied_knobs['message_size_limit']}")
    
    return {
        "config_name": config_name,
        "throughput": throughput,
        "applied_knobs": applied_knobs
    }


def main():
    """主函数"""
    print("="*80)
    print("简化吞吐量测试")
    print("="*80)
    print("\n测试配置:")
    print("1. 默认配置（所有参数默认）")
    print("2. max_inflight_messages=100，其他参数默认")
    print("\n工作负载: 1024B, QoS=1, 10ms")
    print("="*80)
    
    results = []
    
    # 测试配置1: 默认配置
    result1 = test_configuration(
        config_name="默认配置",
        max_inflight_messages=None,
        message_size=1024,
        qos=1,
        publisher_interval_ms=10
    )
    if result1:
        results.append(result1)
    
    # 等待一段时间
    print("\n等待5秒后测试下一个配置...")
    time.sleep(5.0)
    
    # 测试配置2: max_inflight_messages=100
    result2 = test_configuration(
        config_name="max_inflight_100",
        max_inflight_messages=100,
        message_size=1024,
        qos=1,
        publisher_interval_ms=10
    )
    if result2:
        results.append(result2)
    
    # 打印结果汇总
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    for result in results:
        print(f"\n配置: {result['config_name']}")
        print(f"  吞吐量: {result['throughput']:.2f} msg/s")
        print(f"  max_inflight_messages: {result['applied_knobs']['max_inflight_messages']}")
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == "__main__":
    main()
