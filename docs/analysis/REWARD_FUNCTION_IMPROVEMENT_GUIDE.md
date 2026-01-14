# 奖励函数改进指南

## 概述

本文档提供了改进奖励函数的实施指南，包括状态空间扩展、奖励函数重新设计、测试验证等步骤。

## 1. 实施步骤

### 步骤1：扩展状态空间

1. **更新环境配置**
   ```python
   # environment/config.py
   state_dim: int = 10  # 从5扩展到10
   ```

2. **修改状态向量构建函数**
   ```python
   # environment/utils.py
   def build_state_vector(..., latency_p50=0.0, latency_p95=0.0, ...):
       # 添加延迟指标和历史信息
   ```

3. **更新环境类**
   ```python
   # environment/broker.py
   self._throughput_history = []
   self._latency_history = []
   self._history_window = 5
   ```

### 步骤2：实现延迟指标

**临时方案**（当前实现）：
```python
# 使用默认值
latency_p50 = 10.0  # 毫秒
latency_p95 = 50.0  # 毫秒
```

**长期方案**（建议实现）：
```python
# 在WorkloadManager中添加延迟测量
class WorkloadManager:
    def measure_latency(self):
        """测量端到端延迟"""
        # 实现延迟测量逻辑
        return p50_latency, p95_latency
```

### 步骤3：重新设计奖励函数

**核心改进**：
```python
def _compute_reward(self, prev_state, next_state):
    # 1. 绝对性能奖励
    throughput_abs = self._extract_throughput(next_state)
    latency_abs = self._extract_latency(next_state)

    # 2. 相对改进奖励
    throughput_improvement = throughput_abs - prev_throughput
    latency_improvement = prev_latency - latency_abs

    # 3. 稳定性惩罚
    stability_penalty = -2.0 * abs(throughput_improvement + latency_improvement)

    # 4. 综合奖励
    reward = (
        100.0 * throughput_abs +      # 绝对吞吐量
        -50.0 * latency_abs +         # 绝对延迟惩罚
        50.0 * throughput_improvement +  # 吞吐量改进
        30.0 * latency_improvement +     # 延迟改进
        stability_penalty +              # 稳定性惩罚
        resource_penalty                 # 资源约束
    )
    return reward
```

## 2. 权重系数调优

### 当前权重配置
```python
alpha = 100.0   # 绝对吞吐量权重
beta = 50.0     # 绝对延迟权重
gamma = 50.0    # 吞吐量改进权重
delta = 30.0    # 延迟改进权重
epsilon = 1.0   # 稳定性惩罚权重
zeta = 1.0      # 资源惩罚权重
```

### 调优建议

1. **吞吐量 vs 延迟平衡**：
   - 如果强调吞吐量：增加 `alpha`，减少 `beta`
   - 如果强调延迟：减少 `alpha`，增加 `beta`

2. **短期 vs 长期优化**：
   - 如果强调快速改进：增加 `gamma`, `delta`
   - 如果强调整体性能：增加 `alpha`, `beta`

3. **稳定性 vs 探索**：
   - 如果系统不稳定：增加 `epsilon`
   - 如果需要更多探索：减少 `epsilon`

### 调优方法

使用网格搜索或贝叶斯优化：

```python
# 参数范围
param_ranges = {
    'alpha': (50, 200),
    'beta': (20, 100),
    'gamma': (20, 100),
    'delta': (10, 50),
    'epsilon': (0.5, 5.0),
}

# 评估函数
def evaluate_params(alpha, beta, gamma, delta, epsilon):
    # 运行短时间训练，评估收敛速度和最终性能
    return score
```

## 3. 测试验证

### 3.1 单元测试

```python
def test_reward_function():
    """测试奖励函数"""
    env = MosquittoBrokerEnv()

    # 高性能状态（高吞吐量，低延迟）
    high_perf_state = np.array([0.1, 0.8, 0.5, 0.5, 0.1, 0.1, 0.2, 0.0, 0.8, 0.1])

    # 低性能状态（低吞吐量，高延迟）
    low_perf_state = np.array([0.1, 0.2, 0.5, 0.5, 0.1, 0.5, 0.8, 0.0, 0.2, 0.5])

    reward_high = env._compute_reward(None, high_perf_state)
    reward_low = env._compute_reward(None, low_perf_state)

    assert reward_high > reward_low, "高性能状态应获得更高奖励"
    print(f"高性能奖励: {reward_high:.3f}, 低性能奖励: {reward_low:.3f}")
```

### 3.2 训练测试

1. **短时训练测试**：
   ```bash
   # 运行短时间训练，观察奖励曲线
   ./script/run_train.sh --total-timesteps 5000 --save-dir ./test_reward
   ```

2. **对比测试**：
   ```bash
   # 对比新旧奖励函数
   # 旧版本 vs 新版本
   ```

### 3.3 指标监控

**观察的关键指标**：
1. **奖励稳定性**：奖励是否稳定增长，而不是剧烈波动
2. **吞吐量提升**：是否能找到高吞吐量配置
3. **延迟控制**：延迟是否保持在合理范围内
4. **资源使用**：CPU/内存使用是否合理

## 4. 故障排除

### 问题1：奖励值为NaN
```python
# 检查状态向量是否包含NaN
if np.any(np.isnan(next_state)):
    print("警告：状态向量包含NaN值")
    return 0.0  # 返回默认奖励
```

### 问题2：奖励波动过大
```python
# 增加奖励平滑
smoothed_reward = 0.9 * smoothed_reward + 0.1 * raw_reward
```

### 问题3：收敛太慢
```python
# 调整学习率或增加探索
learning_rate = 1e-3  # 提高学习率
action_noise = 0.3    # 增加探索噪声
```

## 5. 性能优化

### 5.1 计算效率

```python
# 缓存历史计算
if not hasattr(self, '_reward_cache'):
    self._reward_cache = {}

cache_key = hash(str(next_state))
if cache_key in self._reward_cache:
    return self._reward_cache[cache_key]
```

### 5.2 内存管理

```python
# 限制历史记录长度
if len(self._throughput_history) > 100:
    self._throughput_history = self._throughput_history[-50:]
```

## 6. 部署和监控

### 6.1 配置备份

```bash
# 备份原有配置
cp environment/config.py environment/config.py.backup
cp environment/broker.py environment/broker.py.backup
```

### 6.2 回滚计划

```bash
# 如果新奖励函数有问题，可以快速回滚
git checkout HEAD~1 environment/
```

### 6.3 监控告警

```python
# 添加训练监控
def monitor_training():
    if reward < -1000:  # 异常奖励阈值
        alert("训练异常：奖励过低")
    if np.std(rewards[-10:]) > 100:  # 奖励波动过大
        alert("训练不稳定：奖励波动过大")
```

## 7. 扩展计划

### 7.1 多目标优化

```python
# 未来可以扩展为多目标RL
objectives = {
    'throughput': throughput_score,
    'latency': -latency_score,
    'stability': stability_score,
    'resource': resource_score
}
```

### 7.2 自适应权重

```python
# 根据训练阶段动态调整权重
if episode < 100:
    # 早期：强调探索
    alpha, beta = 50, 20
else:
    # 后期：强调性能
    alpha, beta = 150, 80
```

## 8. 总结

### 实施清单

- [x] 扩展状态空间到10维
- [x] 添加延迟指标（临时方案）
- [x] 重新设计奖励函数
- [x] 添加稳定性惩罚
- [ ] 实现真实延迟测量
- [ ] 调优权重系数
- [ ] 进行训练测试
- [ ] 监控训练效果

### 预期收益

1. **更稳定的训练**：绝对+相对奖励组合减少波动
2. **更好的性能**：同时优化吞吐量和延迟
3. **更强的泛化**：稳定性约束提高泛化能力
4. **更丰富的学习信号**：多维度奖励提供更多学习信息

### 风险控制

1. **渐进式部署**：先小规模测试，再逐步推广
2. **监控告警**：实时监控训练状态
3. **快速回滚**：准备回滚方案

---

**文档版本**: v1.0
**最后更新**: 2025-01-11
**状态**: 实施中