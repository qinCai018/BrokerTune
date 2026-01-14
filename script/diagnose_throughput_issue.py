#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断吞吐量为0的问题

检查：
1. Broker是否运行
2. Broker是否发布$SYS主题
3. MQTTSampler是否能收到消息
4. 工作负载是否运行
"""

import sys
import time
import subprocess
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environment import EnvConfig
from environment.utils import MQTTSampler

def check_broker_status():
    """检查Broker状态"""
    print("=" * 80)
    print("1. 检查Broker状态")
    print("=" * 80)
    
    # 检查服务状态
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "mosquitto"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            print(f"✅ Broker服务状态: {result.stdout.strip()}")
        else:
            print(f"❌ Broker服务未运行: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 检查服务状态失败: {e}")
        return False
    
    # 检查端口监听
    try:
        result = subprocess.run(
            ["netstat", "-tln"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if ":1883" in result.stdout:
            print("✅ 端口1883正在监听")
        else:
            print("❌ 端口1883未监听")
            return False
    except Exception:
        try:
            result = subprocess.run(
                ["ss", "-tln"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if ":1883" in result.stdout:
                print("✅ 端口1883正在监听")
            else:
                print("❌ 端口1883未监听")
                return False
        except Exception as e:
            print(f"⚠️  无法检查端口监听: {e}")
    
    return True

def check_sys_topics():
    """检查Broker是否发布$SYS主题"""
    print("\n" + "=" * 80)
    print("2. 检查Broker $SYS主题")
    print("=" * 80)
    
    cfg = EnvConfig()
    sampler = None
    
    try:
        print(f"连接到Broker: {cfg.mqtt.host}:{cfg.mqtt.port}")
        sampler = MQTTSampler(cfg.mqtt)
        print("✅ MQTT连接成功")
        
        # 等待连接建立
        time.sleep(1)
        
        # 采样指标（等待更长时间）
        print(f"采样Broker指标（等待{cfg.mqtt.timeout_sec * 2}秒）...")
        metrics = sampler.sample(timeout_sec=cfg.mqtt.timeout_sec * 2)
        
        print(f"\n收到 {len(metrics)} 条指标:")
        if len(metrics) == 0:
            print("❌ 未收到任何$SYS主题消息")
            print("\n可能的原因:")
            print("1. Broker未配置sys_interval（不发布$SYS主题）")
            print("2. Broker刚重启，$SYS主题还未发布")
            print("3. MQTT连接有问题")
        else:
            print("✅ 收到$SYS主题消息:")
            for topic, value in list(metrics.items())[:10]:  # 只显示前10个
                print(f"  {topic}: {value}")
            
            # 检查关键指标
            messages_received = metrics.get("$SYS/broker/messages/received", None)
            clients_connected = metrics.get("$SYS/broker/clients/connected", None)
            
            print(f"\n关键指标:")
            if messages_received is not None:
                print(f"✅ $SYS/broker/messages/received: {messages_received}")
            else:
                print(f"❌ $SYS/broker/messages/received: 未找到")
            
            if clients_connected is not None:
                print(f"✅ $SYS/broker/clients/connected: {clients_connected}")
            else:
                print(f"❌ $SYS/broker/clients/connected: 未找到")
        
        return len(metrics) > 0
        
    except Exception as e:
        print(f"❌ 检查$SYS主题失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if sampler:
            sampler.close()

def check_workload():
    """检查工作负载是否运行"""
    print("\n" + "=" * 80)
    print("3. 检查工作负载")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            ["pgrep", "-f", "emqtt_bench"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"✅ 工作负载正在运行（{len(pids)}个进程）")
            print(f"   PIDs: {', '.join(pids)}")
            return True
        else:
            print("❌ 工作负载未运行")
            print("   没有找到emqtt_bench进程")
            return False
    except Exception as e:
        print(f"❌ 检查工作负载失败: {e}")
        return False

def check_sys_interval_config():
    """检查Broker sys_interval配置"""
    print("\n" + "=" * 80)
    print("4. 检查Broker sys_interval配置")
    print("=" * 80)
    
    config_file = "/etc/mosquitto/mosquitto.conf"
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        if "sys_interval" in content:
            lines = [line for line in content.split('\n') if 'sys_interval' in line.lower()]
            print("✅ 找到sys_interval配置:")
            for line in lines:
                print(f"   {line.strip()}")
            return True
        else:
            print("❌ 未找到sys_interval配置")
            print("   Broker可能不会发布$SYS主题")
            print("\n解决方案:")
            print("   在/etc/mosquitto/mosquitto.conf中添加:")
            print("   sys_interval 10")
            return False
    except Exception as e:
        print(f"⚠️  无法读取配置文件: {e}")
        return False

def main():
    print("=" * 80)
    print("吞吐量为0问题诊断")
    print("=" * 80)
    
    # 1. 检查Broker状态
    broker_ok = check_broker_status()
    if not broker_ok:
        print("\n❌ Broker未正常运行，请先修复Broker问题")
        return
    
    # 2. 检查sys_interval配置
    sys_interval_ok = check_sys_interval_config()
    
    # 3. 检查$SYS主题
    sys_topics_ok = check_sys_topics()
    
    # 4. 检查工作负载
    workload_ok = check_workload()
    
    # 总结
    print("\n" + "=" * 80)
    print("诊断总结")
    print("=" * 80)
    print(f"Broker状态: {'✅' if broker_ok else '❌'}")
    print(f"sys_interval配置: {'✅' if sys_interval_ok else '❌'}")
    print(f"$SYS主题可用: {'✅' if sys_topics_ok else '❌'}")
    print(f"工作负载运行: {'✅' if workload_ok else '❌'}")
    
    print("\n可能的问题:")
    if not sys_interval_ok:
        print("1. ❌ Broker未配置sys_interval，不会发布$SYS主题")
        print("   解决: 在/etc/mosquitto/mosquitto.conf中添加 'sys_interval 10'")
    
    if not sys_topics_ok:
        print("2. ❌ 无法收到$SYS主题消息")
        if not sys_interval_ok:
            print("   原因: Broker未配置sys_interval")
        else:
            print("   原因: 可能是Broker刚重启，需要等待sys_interval时间")
    
    if not workload_ok:
        print("3. ❌ 工作负载未运行")
        print("   解决: 确保训练脚本使用--enable-workload参数")
    
    if sys_interval_ok and sys_topics_ok and workload_ok:
        print("✅ 所有检查通过，如果吞吐量仍为0，可能是:")
        print("   - Broker刚重启，$SYS主题还未发布（等待sys_interval时间）")
        print("   - 工作负载刚启动，还未发送消息")
        print("   - 采样时间太短，未收到消息")

if __name__ == "__main__":
    main()
