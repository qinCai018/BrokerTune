#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 stable_baselines3 的可用导入"""

import sys

print("Python 版本:", sys.version)
print("\n检查 stable_baselines3 导入...")

try:
    import stable_baselines3
    print(f"✓ stable_baselines3 版本: {stable_baselines3.__version__}")
except ImportError as e:
    print(f"✗ 无法导入 stable_baselines3: {e}")
    sys.exit(1)

print("\n检查 DDPG 相关导入...")

# 检查 DDPG
try:
    from stable_baselines3 import DDPG
    print("✓ 可以导入 DDPG")
except ImportError as e:
    print(f"✗ 无法导入 DDPG: {e}")

# 检查 DDPGPolicy
try:
    from stable_baselines3.ddpg.policies import DDPGPolicy
    print("✓ 可以导入 DDPGPolicy from stable_baselines3.ddpg.policies")
except ImportError as e:
    print(f"✗ 无法导入 DDPGPolicy from stable_baselines3.ddpg.policies: {e}")
    
    # 尝试其他路径
    try:
        from stable_baselines3.td3.policies import TD3Policy
        print("✓ 可以导入 TD3Policy from stable_baselines3.td3.policies")
    except ImportError as e2:
        print(f"✗ 无法导入 TD3Policy: {e2}")

# 检查 common.policies
try:
    from stable_baselines3.common.policies import BasePolicy
    print("✓ 可以导入 BasePolicy from stable_baselines3.common.policies")
except ImportError as e:
    print(f"✗ 无法导入 BasePolicy: {e}")

# 列出 ddpg 模块的内容
print("\n检查 stable_baselines3.ddpg 模块内容...")
try:
    import stable_baselines3.ddpg
    print("可用属性:", [x for x in dir(stable_baselines3.ddpg) if not x.startswith('_')])
except Exception as e:
    print(f"✗ 错误: {e}")

# 列出 ddpg.policies 模块的内容
print("\n检查 stable_baselines3.ddpg.policies 模块内容...")
try:
    import stable_baselines3.ddpg.policies
    print("可用属性:", [x for x in dir(stable_baselines3.ddpg.policies) if not x.startswith('_')])
except Exception as e:
    print(f"✗ 错误: {e}")
