#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查工作负载工具是否可用
"""

import subprocess
import os
from pathlib import Path

def check_emqtt_bench():
    """检查 emqtt_bench 是否可用"""
    print("=" * 60)
    print("检查 emqtt_bench 工具...")
    print("=" * 60)
    
    # 检查环境变量
    env_path = os.environ.get("EMQTT_BENCH_PATH")
    if env_path:
        print(f"✓ 找到环境变量 EMQTT_BENCH_PATH: {env_path}")
        if Path(env_path).exists():
            print(f"  → 文件存在")
        else:
            print(f"  → 警告: 文件不存在")
    
    # 检查是否在 PATH 中
    try:
        result = subprocess.run(
            ["which", "emqtt_bench"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            bench_path = result.stdout.strip()
            print(f"✓ 在 PATH 中找到 emqtt_bench: {bench_path}")
        else:
            print("✗ 未在 PATH 中找到 emqtt_bench")
    except Exception as e:
        print(f"✗ 检查 PATH 时出错: {e}")
    
    # 尝试运行帮助命令
    print("\n尝试运行 'emqtt_bench --help'...")
    try:
        result = subprocess.run(
            ["emqtt_bench", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ emqtt_bench 可以正常运行")
            print("\n帮助信息:")
            print(result.stdout[:500])  # 只显示前500个字符
        else:
            print(f"✗ emqtt_bench 运行失败 (退出码: {result.returncode})")
            if result.stderr:
                print(f"错误信息: {result.stderr[:500]}")
    except FileNotFoundError:
        print("✗ emqtt_bench 未找到")
        print("\n安装方法:")
        print("  git clone https://github.com/emqx/emqtt-bench.git")
        print("  cd emqtt-bench")
        print("  make")
        print("\n或者设置环境变量:")
        print("  export EMQTT_BENCH_PATH=/path/to/emqtt_bench")
    except Exception as e:
        print(f"✗ 运行出错: {e}")
    
    print("\n" + "=" * 60)
    print("测试发布/订阅命令...")
    print("=" * 60)
    
    # 测试发布命令
    pub_cmd = [
        "emqtt_bench",
        "pub",
        "-h", "127.0.0.1",
        "-p", "1883",
        "-c", "1",
        "-t", "test/topic",
        "-q", "0",
        "-I", "1000",
        "-s", "10",
    ]
    print(f"\n发布命令: {' '.join(pub_cmd)}")
    print("(这不会实际运行，只是显示命令格式)")
    
    # 测试订阅命令
    sub_cmd = [
        "emqtt_bench",
        "sub",
        "-h", "127.0.0.1",
        "-p", "1883",
        "-c", "1",
        "-t", "test/topic",
        "-q", "0",
    ]
    print(f"\n订阅命令: {' '.join(sub_cmd)}")
    print("(这不会实际运行，只是显示命令格式)")

if __name__ == "__main__":
    check_emqtt_bench()
