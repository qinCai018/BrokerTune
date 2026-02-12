# -*- coding: utf-8 -*-
"""
使用 emqtt_bench 工具为 Mosquitto Broker 生成工作负载。

类似于 sysbench 用于 MySQL，emqtt_bench 用于 MQTT Broker 的性能测试。

使用示例：
    from script.workload import WorkloadManager
    
    # 创建工作负载管理器
    workload = WorkloadManager(
        broker_host="127.0.0.1",
        broker_port=1883,
        emqtt_bench_path="./emqtt-bench/emqtt_bench"
    )
    
    # 启动工作负载
    workload.start(
        num_publishers=100,
        num_subscribers=100,
        topic="test/topic",
        message_rate=100,  # 每秒消息数
        message_size=100,  # 消息大小（字节）
        duration=300  # 运行时间（秒）
    )
    
    # ... 进行测试 ...
    
    # 停止工作负载
    workload.stop()
"""

from __future__ import annotations

import subprocess
import time
import os
import signal
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class WorkloadConfig:
    """工作负载配置"""
    # 发布者配置
    num_publishers: int = 100
    publisher_interval_ms: int = 100  # 每个发布者发布消息的间隔（毫秒）
    publisher_messages: int = 0  # 每个发布者发布的消息总数（0表示持续发布）
    
    # 订阅者配置
    num_subscribers: int = 100
    
    # 连接测试配置
    num_connections: int = 0  # 仅连接，不发布/订阅
    
    # 消息配置
    topic: str = "test/topic"
    message_size: int = 100  # 消息大小（字节）
    message_payload: Optional[str] = None  # 自定义消息内容
    
    # QoS 配置
    qos: int = 0  # QoS 级别：0, 1, 或 2
    
    # 运行时间（秒），0表示持续运行直到手动停止
    duration: int = 0


class WorkloadManager:
    """
    管理工作负载的生命周期：启动、监控、停止。
    
    使用 emqtt_bench 工具生成 MQTT 工作负载。
    """
    
    def __init__(
        self,
        broker_host: str = "127.0.0.1",
        broker_port: int = 1883,
        emqtt_bench_path: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        初始化工作负载管理器。
        
        Args:
            broker_host: MQTT Broker 地址
            broker_port: MQTT Broker 端口
            emqtt_bench_path: emqtt_bench 可执行文件路径
                            如果为 None，会尝试从环境变量或默认路径查找
            username: MQTT 用户名（可选）
            password: MQTT 密码（可选）
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        
        # 查找 emqtt_bench 路径
        if emqtt_bench_path is None:
            emqtt_bench_path = os.environ.get("EMQTT_BENCH_PATH", "emqtt_bench")
        
        self.emqtt_bench_path = Path(emqtt_bench_path)
        if not self.emqtt_bench_path.exists() and not self._is_in_path(emqtt_bench_path):
            raise FileNotFoundError(
                f"emqtt_bench 未找到: {emqtt_bench_path}\n"
                f"请安装 emqtt_bench 或设置 EMQTT_BENCH_PATH 环境变量\n"
                f"安装方法: git clone https://github.com/emqx/emqtt-bench.git && cd emqtt-bench && make"
            )
        
        self.emqtt_bench_cmd = str(self.emqtt_bench_path) if self.emqtt_bench_path.exists() else emqtt_bench_path
        
        # 存储运行中的进程
        self._processes: List[subprocess.Popen] = []
        self._is_running = False
        self._last_config: Optional[WorkloadConfig] = None  # 保存最后一次使用的配置，用于重启
    
    def _is_in_path(self, cmd: str) -> bool:
        """检查命令是否在系统 PATH 中"""
        try:
            subprocess.run(
                ["which", cmd],
                capture_output=True,
                check=True,
                timeout=1
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def start(
        self,
        config: Optional[WorkloadConfig] = None,
        num_publishers: int = 100,
        num_subscribers: int = 100,
        topic: str = "test/topic",
        message_rate: int = 100,
        message_size: int = 100,
        duration: int = 0,
        qos: int = 0,
    ) -> None:
        """
        启动工作负载。
        
        Args:
            config: 工作负载配置对象（如果提供，会覆盖其他参数）
            num_publishers: 发布者数量
            num_subscribers: 订阅者数量
            topic: MQTT 主题
            message_rate: 每秒消息数（所有发布者总计）
            message_size: 消息大小（字节）
            duration: 运行时间（秒），0表示持续运行
            qos: QoS 级别（0, 1, 或 2）
        """
        # 如果已经在运行，先停止
        if self._is_running:
            print("[工作负载] 检测到工作负载已在运行，先停止旧进程...")
            self.stop()
        
        if config is None:
            config = WorkloadConfig(
                num_publishers=num_publishers,
                num_subscribers=num_subscribers,
                topic=topic,
                message_size=message_size,
                qos=qos,
                duration=duration,
            )
            # 计算每个发布者的间隔（毫秒）
            if num_publishers > 0 and message_rate > 0:
                config.publisher_interval_ms = max(1, int(1000 * num_publishers / message_rate))
        
        self._processes = []
        
        # 保存配置用于后续重启
        self._last_config = config
        
        # 启动订阅者
        if config.num_subscribers > 0:
            sub_cmd = self._build_sub_command(config)
            print(f"[工作负载] 执行订阅命令: {' '.join(sub_cmd)}")
            sub_process = subprocess.Popen(
                sub_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None,
            )
            self._processes.append(sub_process)
            print(f"[工作负载] 启动 {config.num_subscribers} 个订阅者 (PID: {sub_process.pid})...")
            time.sleep(1)  # 等待订阅者连接
            
            # 检查进程是否启动成功
            if sub_process.poll() is not None:
                # 进程已退出，读取错误信息
                stdout, stderr = sub_process.communicate()
                stdout_msg = stdout.decode('utf-8', errors='ignore') if stdout else ""
                stderr_msg = stderr.decode('utf-8', errors='ignore') if stderr else ""
                
                error_info = []
                if stdout_msg:
                    error_info.append(f"标准输出: {stdout_msg[:500]}")
                if stderr_msg:
                    error_info.append(f"错误输出: {stderr_msg[:500]}")
                if not error_info:
                    error_info.append("无输出信息")
                
                raise RuntimeError(
                    f"订阅者进程启动失败 (退出码: {sub_process.returncode})\n"
                    f"命令: {' '.join(sub_cmd)}\n"
                    + "\n".join(error_info)
                )
        
        # 启动发布者
        if config.num_publishers > 0:
            pub_cmd = self._build_pub_command(config)
            print(f"[工作负载] 执行发布命令: {' '.join(pub_cmd)}")
            pub_process = subprocess.Popen(
                pub_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None,
            )
            self._processes.append(pub_process)
            print(f"[工作负载] 启动 {config.num_publishers} 个发布者 (PID: {pub_process.pid})...")
            
            # 检查进程是否启动成功
            time.sleep(0.5)  # 短暂等待，检查进程是否立即退出
            if pub_process.poll() is not None:
                # 进程已退出，读取错误信息
                stdout, stderr = pub_process.communicate()
                stdout_msg = stdout.decode('utf-8', errors='ignore') if stdout else ""
                stderr_msg = stderr.decode('utf-8', errors='ignore') if stderr else ""
                
                error_info = []
                if stdout_msg:
                    error_info.append(f"标准输出: {stdout_msg[:500]}")
                if stderr_msg:
                    error_info.append(f"错误输出: {stderr_msg[:500]}")
                if not error_info:
                    error_info.append("无输出信息")
                
                raise RuntimeError(
                    f"发布者进程启动失败 (退出码: {pub_process.returncode})\n"
                    f"命令: {' '.join(pub_cmd)}\n"
                    + "\n".join(error_info)
                )
        
        # 启动连接测试（如果配置）
        if config.num_connections > 0:
            conn_cmd = self._build_conn_command(config)
            conn_process = subprocess.Popen(
                conn_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None,
            )
            self._processes.append(conn_process)
            print(f"启动 {config.num_connections} 个连接...")
        
        self._is_running = True
        print(f"[工作负载] 工作负载已启动，共 {len(self._processes)} 个进程")
        print(f"[工作负载] 主题: {config.topic}, QoS: {config.qos}")
        print(f"[工作负载] 发布者间隔: {config.publisher_interval_ms}ms")
        
        # 验证工作负载是否真的在发送消息（等待5秒后验证）
        if config.num_publishers > 0:
            print(f"[工作负载] 等待5秒后验证消息发送...")
            time.sleep(5.0)
            if self._verify_messages_sending(config.topic):
                print(f"[工作负载] ✅ 验证成功：工作负载正在发送消息到主题 '{config.topic}'")
            else:
                print(f"[工作负载] ⚠️  警告：无法验证消息发送，但进程仍在运行")
                print(f"[工作负载] 提示：可以使用以下命令手动验证:")
                print(f"  mosquitto_sub -h {self.broker_host} -p {self.broker_port} -t '{config.topic}' -C 1")
        
        # 如果设置了持续时间，在后台启动定时停止
        if config.duration > 0:
            def _auto_stop():
                time.sleep(config.duration)
                if self._is_running:
                    self.stop()
            
            import threading
            timer = threading.Thread(target=_auto_stop, daemon=True)
            timer.start()
    
    def _build_base_args(self, subcommand: str) -> List[str]:
        """构建基础命令行参数
        
        Args:
            subcommand: 子命令（pub/sub/conn），必须放在命令开头
        """
        args = [
            self.emqtt_bench_cmd,
            subcommand,  # 子命令必须在最前面
            "-h", self.broker_host,
            "-p", str(self.broker_port),
        ]
        
        if self.username:
            args.extend(["-u", self.username])
        if self.password:
            args.extend(["-P", self.password])
        
        return args
    
    def _build_pub_command(self, config: WorkloadConfig) -> List[str]:
        """构建发布命令"""
        cmd = self._build_base_args("pub") + [
            "-c", str(config.num_publishers),
            "-t", config.topic,
            "-q", str(config.qos),
            "-I", str(config.publisher_interval_ms),  # 消息发布间隔（毫秒）
            "-s", str(config.message_size),  # 消息大小（字节）
        ]
        
        if config.publisher_messages > 0:
            cmd.extend(["-n", str(config.publisher_messages)])
        
        # 如果指定了自定义消息内容，使用 -m 参数（注意：某些版本的 emqtt_bench 可能不支持 -m）
        if config.message_payload:
            cmd.extend(["-m", config.message_payload])
        
        return cmd
    
    def _build_sub_command(self, config: WorkloadConfig) -> List[str]:
        """构建订阅命令"""
        cmd = self._build_base_args("sub") + [
            "-c", str(config.num_subscribers),
            "-t", config.topic,
            "-q", str(config.qos),
        ]
        return cmd
    
    def _build_conn_command(self, config: WorkloadConfig) -> List[str]:
        """构建连接命令"""
        cmd = self._build_base_args("conn") + [
            "-c", str(config.num_connections),
        ]
        return cmd

    def get_latency_probe_debug(self) -> Dict[str, Any]:
        """主分支未实现延迟探测，返回空调试信息。"""
        return {
            "connected": False,
            "samples": 0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "topic": "",
            "interval_sec": 0.0,
            "window_size": 0,
        }
    
    def stop(self) -> None:
        """停止所有工作负载进程"""
        if not self._is_running:
            return
        
        print("正在停止工作负载...")
        
        for process in self._processes:
            try:
                if os.name != 'nt':
                    # Unix/Linux: 发送信号到进程组
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    # Windows: 终止进程
                    process.terminate()
                
                # 等待进程结束，最多等待5秒
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # 强制终止
                    if os.name != 'nt':
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            except (ProcessLookupError, OSError) as e:
                # 进程可能已经结束
                print(f"警告: 停止进程时出错: {e}")
        
        self._processes = []
        self._is_running = False
        print("工作负载已停止")
    
    def is_running(self) -> bool:
        """检查工作负载是否正在运行"""
        if not self._is_running:
            return False
        
        # 检查进程是否还在运行
        alive_processes = []
        for process in self._processes:
            if process.poll() is None:  # 进程仍在运行
                alive_processes.append(process)
            else:
                # 进程已结束
                print(f"[工作负载] 警告: 工作负载进程已结束 (退出码: {process.returncode})")
        
        self._processes = alive_processes
        self._is_running = len(self._processes) > 0
        
        return self._is_running
    
    def restart(self) -> None:
        """
        重启工作负载（使用最后一次的配置）
        
        如果工作负载已停止，会使用保存的配置重新启动。
        如果工作负载正在运行，会先停止再重启。
        """
        if self._last_config is None:
            raise RuntimeError("无法重启：没有保存的配置。请先调用 start() 启动工作负载。")
        
        print("[工作负载] 重启工作负载...")
        if self._is_running:
            self.stop()
        
        # 等待进程完全停止
        time.sleep(1.0)
        
        # 使用保存的配置重启
        self.start(config=self._last_config)
    
    def _verify_messages_sending(self, topic: str, timeout_sec: float = 5.0) -> bool:
        """
        验证工作负载是否真的在发送消息
        
        Args:
            topic: MQTT主题
            timeout_sec: 超时时间（秒）
            
        Returns:
            True如果收到消息，False如果未收到
        """
        try:
            import paho.mqtt.client as mqtt
            received_messages = []
            
            def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    client.subscribe(topic)
                else:
                    received_messages.append(None)  # 标记连接失败
            
            def on_message(client, userdata, msg):
                received_messages.append(msg.payload)
                client.disconnect()  # 收到一条消息就断开
            
            client = mqtt.Client()
            client.on_connect = on_connect
            client.on_message = on_message
            
            # 连接到Broker
            try:
                client.connect(self.broker_host, self.broker_port, 60)
                client.loop_start()
                
                # 等待消息
                start_time = time.time()
                while len(received_messages) == 0 and (time.time() - start_time) < timeout_sec:
                    time.sleep(0.1)
                
                client.loop_stop()
                client.disconnect()
                
                return len(received_messages) > 0 and received_messages[0] is not None
            except Exception as e:
                print(f"[工作负载] 验证消息发送时出错: {e}")
                return False
        except ImportError:
            # paho-mqtt未安装，跳过验证
            print("[工作负载] 警告: paho-mqtt未安装，跳过消息验证")
            return True  # 假设成功，避免阻塞
        except Exception as e:
            print(f"[工作负载] 验证消息发送时出错: {e}")
            return False
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行 MQTT 工作负载")
    parser.add_argument(
        "--broker-host",
        type=str,
        default="127.0.0.1",
        help="MQTT Broker 地址（默认：127.0.0.1）",
    )
    parser.add_argument(
        "--broker-port",
        type=int,
        default=1883,
        help="MQTT Broker 端口（默认：1883）",
    )
    parser.add_argument(
        "--emqtt-bench-path",
        type=str,
        default=None,
        help="emqtt_bench 可执行文件路径",
    )
    parser.add_argument(
        "--publishers",
        type=int,
        default=100,
        help="发布者数量（默认：100）",
    )
    parser.add_argument(
        "--subscribers",
        type=int,
        default=100,
        help="订阅者数量（默认：100）",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="test/topic",
        help="MQTT 主题（默认：test/topic）",
    )
    parser.add_argument(
        "--message-rate",
        type=int,
        default=100,
        help="每秒消息数（默认：100）",
    )
    parser.add_argument(
        "--message-size",
        type=int,
        default=100,
        help="消息大小（字节，默认：100）",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="运行时间（秒，0表示持续运行直到手动停止，默认：0）",
    )
    parser.add_argument(
        "--qos",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="QoS 级别（默认：0）",
    )
    
    args = parser.parse_args()
    
    workload = WorkloadManager(
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        emqtt_bench_path=args.emqtt_bench_path,
    )
    
    try:
        workload.start(
            num_publishers=args.publishers,
            num_subscribers=args.subscribers,
            topic=args.topic,
            message_rate=args.message_rate,
            message_size=args.message_size,
            duration=args.duration,
            qos=args.qos,
        )
        
        if args.duration == 0:
            print("工作负载正在运行，按 Ctrl+C 停止...")
            try:
                while workload.is_running():
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n收到停止信号...")
        else:
            print(f"工作负载将运行 {args.duration} 秒...")
            while workload.is_running():
                time.sleep(1)
    finally:
        workload.stop()
