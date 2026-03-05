# N-step / PER 面向 Mosquitto BrokerTune 的最小侵入适配说明

## 1. 为什么要改（问题背景）
BrokerTune 的单步交互成本高（一个动作可能触发 Broker 重启与稳定等待），同时训练过程存在非平稳漂移（负载变化、重启后分布切换）与时延约束（`lagrangian_hinge`）。

在这种场景下：
- 经验样本珍贵，N-step 需要避免跨 episode/跨重启拼接；
- PER 不能长期被历史极端 TD-error 支配；
- 约束相关样本需要“可控地”提高被关注概率，但默认行为必须兼容旧训练脚本。

## 2. 代码审计基线（改动前后的真实入口）

### 2.1 N-step / PER 核心实现与训练调用链
- 回放缓冲实现：`model/prioritized_nstep_replay_buffer.py`（当前主实现区：`96-648`）。
- DDPG 训练中使用 N-step 折扣与 PER 回写：`model/enhanced_ddpg.py:19-130`。
- Replay buffer 注入入口：`tuner/utils.py:153-188`。
- CLI 开关入口：`tuner/train.py:183-279`，并在 `tuner/train.py:1473-1504` 传入模型。
- Replay 调试日志：`tuner/train.py:1137-1167`（`ReplayDebugCallback` 调 `get_debug_stats()`）。

### 2.2 done / reward / constraint / reset 语义
- step 返回：`environment/broker.py:483-523`，语义为 `(obs, reward, terminated, truncated, info)`。
- 失败路径：`environment/broker.py:525-562`，`truncated` 由连续失败阈值触发。
- 约束信号来源：`environment/broker.py:786-826`（`reward_components` 含 `latency_violation_ms`、`latency_limit_ms`、`unsafe`）。
- 重启节奏：`environment/broker.py:281-358`（动作变化触发 Broker 重启，伴随等待与 workload 恢复）。

## 3. 关键公式（实现口径）

### 3.1 N-step 目标
```math
G_t^{(n)} = \sum_{k=0}^{n_t-1} \gamma^k r_{t+k}
```

```math
y_t = G_t^{(n)} + (1-d_t)\gamma^{n_t} Q_{\text{target}}\bigl(s_{t+n_t}, \pi_{\text{target}}(s_{t+n_t})\bigr)
```

其中 $n_t$ 为样本真实有效步数（可能小于配置 `n_step`），$d_t$ 为统一 done 信号。

### 3.2 约束调制 priority
```math
u_i = (|\delta_i| + \epsilon) \cdot w_i
```

```math
w_i = 1 + \lambda_c \cdot \frac{\max(0,\text{latency\_violation\_ms})}{\max(\text{latency\_limit\_ms},\epsilon)}
```

```math
u_i \leftarrow \min(u_i, p_{\text{clip\_max}}),\quad u_i \leftarrow \max(u_i,\epsilon),\quad p_i=u_i^{\alpha}
```

当 `per_constraint_priority=False` 时，$w_i=1$（与旧行为兼容）。

## 4. 本次改动（逐条记录）

### 4.1 N-step 相关改动

- ==context==统一 done 语义：在 `_extract_single_transition` 中使用 `done_array OR info.done OR info.terminated OR info.truncated`，避免包装层语义不一致。==  
  代码：`model/prioritized_nstep_replay_buffer.py:223-250`。  
  影响范围：仅回放缓冲内部 done 判定。  
  默认开关：无（始终生效）。  
  兼容性：向后兼容（旧 `done` 路径保留）。

- ==context==新增跨边界保护：基于 `info["step"]` 检测不连续（`<=last` 或 `>last+1`）时，强制 flush 当前 N-step 队列并以 terminal 方式截断，避免重启前后串联。==  
  代码：`model/prioritized_nstep_replay_buffer.py:395-405`。  
  影响范围：仅 N-step 聚合边界。  
  默认开关：无（自动生效）。  
  兼容性：不改变默认训练接口，仅提高边界鲁棒性。

- ==context==N-step 聚合接口增强：`_build_n_step_from_queue(max_steps, force_done)` 支持显式截断与强制 terminal。==  
  代码：`model/prioritized_nstep_replay_buffer.py:281-327`。  
  影响范围：N-step return 的构造细节。  
  默认开关：无。  
  兼容性：保持原奖励折扣逻辑，仅扩展边界控制能力。

- ==context==新增可选 `n_step_adaptive`：基于最近窗口 done 比例，当 done 比例超过阈值时有效步长自动降为 `max(1, n_step//2)`。==  
  代码：`model/prioritized_nstep_replay_buffer.py:199-221`、`406-414`。  
  影响范围：仅启用开关时改变有效 `n`。  
  默认开关：`False`。  
  兼容性：默认完全保持旧行为。

### 4.2 PER 相关改动

- ==context==扩展 PER 参数：`per_beta_end`、`per_clip_max`、`per_mix_uniform_ratio`、`per_constraint_priority`、`per_constraint_scale`。==  
  代码：`model/prioritized_nstep_replay_buffer.py:103-147`、`tuner/utils.py:59-172`、`tuner/train.py:197-279`。  
  影响范围：参数层与回放缓冲采样/更新策略。  
  默认开关：均保持兼容（`beta_end=1.0`，其余开关默认关闭）。  
  兼容性：旧参数与旧命令行可继续使用。

- ==context==beta 调度改为 `beta_start -> beta_end` 线性退火（按采样步计数），并保留 IS 权重归一化与 eps 防护。==  
  代码：`model/prioritized_nstep_replay_buffer.py:464-541`。  
  影响范围：PER 采样权重稳定性。  
  默认开关：默认 `beta_start=0.4, beta_end=1.0`。  
  兼容性：退火终点与旧逻辑一致（1.0）。

- ==context==新增 priority clip/floor 与可选约束调制：`update_priorities(indices, td_errors, constraint_signals=None)`。==  
  代码：`model/prioritized_nstep_replay_buffer.py:574-618`。  
  影响范围：priority 更新阶段。  
  默认开关：`per_constraint_priority=False`、`per_clip_max=0`（不启用裁剪）。  
  兼容性：保留旧签名调用路径，`constraint_signals` 可选。

- ==context==新增可选 mix sampling：PER 样本与均匀样本按比例混采，uniform 样本 IS 权重固定为 1。==  
  代码：`model/prioritized_nstep_replay_buffer.py:479-541`。  
  影响范围：采样分布（仅开关开启时）。  
  默认开关：`per_mix_uniform_ratio=0`（关闭）。  
  兼容性：默认与旧采样一致。

### 4.3 训练与环境配套改动

- ==context==训练侧 priority 回写支持 `constraint_ratios`，并新增 `is_weight_min/max`、`per_constraint_ratio` 日志。==  
  代码：`model/enhanced_ddpg.py:71-130`。  
  影响范围：logger 与 PER 回写参数。  
  默认开关：约束调制关闭时仅记录，不改变策略。  
  兼容性：不改 TD 目标主流程。

- ==context==环境 `info` 显式补充 `terminated`、`truncated`、`done` 字段，统一上游可读语义。==  
  代码：`environment/broker.py:491-517`、`543-556`。  
  影响范围：transition info 字段。  
  默认开关：无。  
  兼容性：仅新增字段，不破坏旧字段。

## 5. 新增/调整参数清单（默认行为）

### 5.1 CLI（`tuner/train.py`）
- `--per_beta_start`（别名复用 `--per_beta0`，默认 `0.4`）
- `--per_beta_end`（默认 `1.0`）
- `--per_clip_max`（默认 `0.0`，表示关闭）
- `--per_mix_uniform_ratio`（默认 `0.0`）
- `--per_constraint_priority`（默认 `0`）
- `--per_constraint_scale`（默认 `1.0`）
- `--n_step_adaptive`（默认 `0`）

### 5.2 Model/replay kwargs 透传
- `tuner/utils.py:155-172` 将上述参数传入 `PrioritizedNStepReplayBuffer`。

## 6. 可观测性（Replay Debug / Train 日志）

### 6.1 `get_debug_stats()` 新增指标
代码：`model/prioritized_nstep_replay_buffer.py:623-648`
- `nstep_effective_n`
- `nstep_done_ratio`
- `nstep_forced_flushes`
- `per_last_priority_min/mean/p95/max`
- `per_last_weight_min/mean/max`
- `per_last_constraint_ratio`

### 6.2 `EnhancedDDPG` 新增训练日志
代码：`model/enhanced_ddpg.py:119-130`
- `train/is_weight_min`
- `train/is_weight_mean`
- `train/is_weight_max`
- `train/per_constraint_ratio`

## 7. 回归测试与结果

### 7.1 新增/扩展测试
- `tests/test_replay_buffer_per_nstep.py`
  - done 统一语义（terminated/truncated）
  - step 不连续强制 flush
  - adaptive n-step 触发
  - beta start/end 调度
  - priority clip/floor + constraint modulation 单调性
  - mix sampling 与 debug stats
- `tests/test_env_reward.py`
  - 失败转移 info 的 `terminated/truncated/done` 字段断言

### 7.2 执行命令
```bash
PYTHONPATH=$(pwd):$PYTHONPATH pytest -q tests/test_replay_buffer_per_nstep.py tests/test_env_reward.py tests/test_attention_extractor.py
```

本次回归结果：`16 passed`。

## 8. 兼容性声明
- 默认配置下，新增能力均关闭或退化到旧行为，不改变既有训练命令的基本语义。
- 旧参数 `--per_beta0/--per-beta0` 继续可用，并作为 `beta_start`。
- `update_priorities` 保持旧调用兼容；`constraint_signals` 仅为可选扩展。
- 环境 info 仅新增字段，不删除旧字段。
