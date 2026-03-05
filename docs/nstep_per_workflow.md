# BrokerTune 中 N-step 与 PER 工作流程（Mosquitto 适配版）

## 1. 代码定位总览

1. N-step 与 PER 主实现：`model/prioritized_nstep_replay_buffer.py:96-646`。
2. DDPG 训练循环（采样、TD-error、priority 回写）：`model/enhanced_ddpg.py:19-130`。
3. 模型接线（ReplayBuffer 注入）：`tuner/utils.py:153-188`。
4. CLI 开关（PER/N-step 参数）：`tuner/train.py:183-279`。
5. 训练入口参数透传：`tuner/train.py:1473-1504`。
6. 环境 done 语义来源：`environment/broker.py:483-523` 与 `environment/broker.py:525-562`。
7. 包装层对 4/5 元组 done 兼容：`tuner/train.py:751-757`。
8. Broker 重启与 step 边界语义：`environment/broker.py:281-358`。

## 2. 环境交互与 done/terminated/truncated 语义

在环境 `step()` 中，返回值采用 Gymnasium 五元组 $(s_{t+1}, r_t, \text{terminated}_t, \text{truncated}_t, \text{info}_t)$，并在 `info` 中显式写入 `done = terminated OR truncated`（`environment/broker.py:483-497`）。失败路径 `_make_failure_transition()` 也保持同一语义（`environment/broker.py:540-549`）。

包装层 `ActionThroughputLoggerWrapper.step()` 对 4 元组与 5 元组统一处理（`tuner/train.py:751-757`）：

```math
d_t = \mathbf{1}[\text{terminated}_t \lor \text{truncated}_t]
```

BrokerTune 的单步语义中，动作变化会触发 broker 重启（`environment/broker.py:291-306`），并伴随等待与工作负载恢复（`environment/broker.py:325-357`），因此跨重启拼接轨迹是需要显式防护的边界问题。

## 3. N-step 工作流程（按代码）

### Step 0：transition 输入结构与来源

1. 处理对象：单步 transition $\tau_t=(s_t,a_t,r_t,s_{t+1},d_t,\mathrm{info}_t)$，其中 $s_t\in\mathbb{R}^{d}$，$a_t\in\mathbb{R}^{m}$，$d_t\in\{0,1\}$。
2. 达到的效果：把 collector 传入的 `obs/next_obs/action/reward/done/infos` 统一抽取为 `_NStepTransition`。
3. 数学表达式：

```math
\tau_t = (s_t, a_t, r_t, s_{t+1}, d_t, \mathrm{info}_t)
```

代码定位：`model/prioritized_nstep_replay_buffer.py:223-250`。

### Step 1：done 统一与 episode 边界定义

1. 处理对象：$d_t$ 与 `info` 中的 `done/terminated/truncated`。
2. 达到的效果：在 replay 入口统一 done 语义，防止上游包装差异导致跨 episode 串联。
3. 数学表达式：

```math
d_t = \mathbf{1}[d_t^{(array)} \lor d_t^{(info)} \lor \text{terminated}_t \lor \text{truncated}_t]
```

==context==将 done 统一为 `done_array OR info.done OR info.terminated OR info.truncated`，避免多包装链路语义不一致。==  
代码定位：`model/prioritized_nstep_replay_buffer.py:236-241`。

### Step 2：n-step 队列入队与可输出条件

1. 处理对象：队列 $\mathcal{Q}_t=[\tau_{t-k+1},\dots,\tau_t]$ 与有效步长 $n_t^{\mathrm{eff}}$。
2. 达到的效果：按 `n_step`（或自适应后的有效步长）控制何时聚合输出。
3. 数学表达式：

```math
\text{if } |\mathcal{Q}_t| \ge n_t^{\mathrm{eff}},\ \text{flush one N-step sample}
```

==context==检测 `info["step"]` 不连续（`<=last` 或 `>last+1`）时，强制 flush 并按 terminal 截断，避免 broker 重启/复位前后被拼接为同一 n-step 轨迹。==  
代码定位：`model/prioritized_nstep_replay_buffer.py:395-405`。

==context==支持 `n_step_adaptive`：当最近窗口 done 比例超过阈值时将有效步长降为 `max(1, n_step//2)`，降低高截断场景下长回报估计不稳定。==  
代码定位：`model/prioritized_nstep_replay_buffer.py:199-221`。

### Step 3：n-step return 计算与截断

1. 处理对象：$\mathcal{Q}_t$ 中的奖励序列与 done 标记。
2. 达到的效果：计算实际可用步长 $n_t$ 的折扣回报；遇到 done 或强制截断立即停止。
3. 数学表达式：

```math
G_t^{(n_t)} = \sum_{k=0}^{n_t-1} \gamma^k r_{t+k}
```

```math
n_t = \min\{k\le n_t^{\mathrm{eff}} \mid d_{t+k-1}=1\}\ \text{or}\ n_t^{\mathrm{eff}}
```

代码定位：`model/prioritized_nstep_replay_buffer.py:281-307`。

### Step 4：写入 replay 的 n-step transition 结构

1. 处理对象：输出样本 $(s_t,a_t,G_t^{(n_t)},s_{t+n_t},d_{t+n_t-1},n_t)$。
2. 达到的效果：把 n-step 聚合结果写入 SB3 ReplayBuffer，并单独保存 `n_steps[index]`。
3. 数学表达式：

```math
\tilde{\tau}_t = (s_t, a_t, G_t^{(n_t)}, s_{t+n_t}, d_{t+n_t-1}, n_t)
```

代码定位：`model/prioritized_nstep_replay_buffer.py:252-279`。

### Step 5：与 DDPG target 的衔接

1. 处理对象：训练 batch 中的 $G_t^{(n_t)}$、$d_t$、$n_t$ 与目标网络 $Q_{\text{target}}$。
2. 达到的效果：在训练阶段使用样本内的 `n_steps` 计算 $\gamma^{n_t}$，与 DDPG 目标拼接。
3. 数学表达式：

```math
y_t = G_t^{(n_t)} + (1-d_t)\gamma^{n_t}Q_{\text{target}}(s_{t+n_t},\pi_{\text{target}}(s_{t+n_t}))
```

代码定位：`model/enhanced_ddpg.py:42-57`。

## 4. PER 工作流程（按代码）

### Step 0：新样本 priority 初始化

1. 处理对象：新写入样本索引 $i$ 的优先级 $p_i$。
2. 达到的效果：新样本初始化为当前最大 priority，保证可被采到。
3. 数学表达式：

```math
p_i \leftarrow p_{\max}
```

代码定位：`model/prioritized_nstep_replay_buffer.py:272-277`。

### Step 1：基于 priority 的采样概率（sum-tree）

1. 处理对象：sum-tree 上的 $\{p_i\}$。
2. 达到的效果：按分段前缀和抽样，近似实现按 $p_i$ 比例采样。
3. 数学表达式：

```math
P(i) = \frac{p_i}{\sum_j p_j}
```

代码定位：`model/prioritized_nstep_replay_buffer.py:426-439` 与 `model/prioritized_nstep_replay_buffer.py:513-520`。

### Step 2：IS 权重计算与归一化

1. 处理对象：采样概率 $P(i)$、buffer 大小 $N$、退火系数 $\beta$。
2. 达到的效果：计算 importance sampling 权重并用最大权重归一化，稳定训练。
3. 数学表达式：

```math
w_i = (N\cdot P(i))^{-\beta}
```

```math
\hat{w}_i = \frac{w_i}{\max_j w_j}
```

代码定位：`model/prioritized_nstep_replay_buffer.py:520-533`。

### Step 3：返回训练 batch

1. 处理对象：$(s,a,r,s',d)$、`indices`、`weights`、`n_steps`、`constraint_ratios`。
2. 达到的效果：为训练循环同时提供 PER 权重与 priority 回写索引。
3. 数学表达式：

```math
\mathcal{B} = \{(s_i,a_i,r_i,s'_i,d_i,n_i,\hat{w}_i,\mathrm{idx}_i,c_i)\}_{i=1}^{B}
```

代码定位：`model/prioritized_nstep_replay_buffer.py:562-572`。

### Step 4：训练中的 TD-error 计算

1. 处理对象：critic 输出 $Q_\theta$ 与目标 $y_t$。
2. 达到的效果：得到用于 PER 回写的样本级 TD-error。
3. 数学表达式：

```math
\delta_i = \left|Q_{\theta}(s_i,a_i) - y_i\right|
```

代码定位：`model/enhanced_ddpg.py:72`。

### Step 5：priority 更新规则

1. 处理对象：$\delta_i$、$\epsilon$、$\alpha$、可选约束信号 $c_i$。
2. 达到的效果：按 TD-error 为主更新 priority，并可叠加约束调制与截断。
3. 数学表达式：

```math
u_i = (|\delta_i| + \epsilon)\cdot w_i^{\mathrm{constraint}}
```

```math
w_i^{\mathrm{constraint}} = 1 + \lambda_c c_i
```

```math
u_i \leftarrow \max(\epsilon,\min(u_i,p_{\mathrm{clip\_max}})),\quad p_i \leftarrow u_i^{\alpha}
```

==context==加入 `per_clip_max` 上限，防止非平稳场景下极端 TD-error 长期霸榜。==  
代码定位：`model/prioritized_nstep_replay_buffer.py:607-609`。

==context==加入可开关约束调制 `per_constraint_priority`，并用 `latency_violation_ms/latency_limit_ms`（或 `unsafe` 回退）构造约束比率。==  
代码定位：`model/prioritized_nstep_replay_buffer.py:328-358`、`model/prioritized_nstep_replay_buffer.py:597-603`。

### Step 6：$\alpha/\beta$ 调度

1. 处理对象：$\alpha$ 与 $\beta$。
2. 达到的效果：$\alpha$ 固定，$\beta$ 线性从 `per_beta0` 退火到 `per_beta_end`。
3. 数学表达式：

```math
\beta_t = \beta_{\mathrm{start}} + \rho_t(\beta_{\mathrm{end}}-\beta_{\mathrm{start}}),\quad \rho_t\in[0,1]
```

```math
\alpha_t = \alpha\ \text{(constant)}
```

代码定位：`model/prioritized_nstep_replay_buffer.py:464-469`（beta 调度）、`model/prioritized_nstep_replay_buffer.py:611`（alpha 使用，未调度）。

==context==支持 mix sampling（PER + 均匀采样混合），其中均匀采样子集 IS 权重固定为 1，用于缓解 PER 偏置。==  
代码定位：`model/prioritized_nstep_replay_buffer.py:482-497`、`model/prioritized_nstep_replay_buffer.py:517-533`。

## 5. Mosquitto/BrokerTune 适配改动清单（高亮）

==context==N-step 在检测到 `info["step"]` 断点时强制 flush 并 terminal 截断，防止 broker 重启或 reset 前后跨边界串联。==  
代码定位：`model/prioritized_nstep_replay_buffer.py:395-405`。

==context==N-step 的 done 统一为四路 OR（array/info/terminated/truncated），避免 collector/wrapper 语义差异。==  
代码定位：`model/prioritized_nstep_replay_buffer.py:236-241`。

==context==可选 `n_step_adaptive` 根据 done 比例动态减半有效步长，降低高失败/高截断阶段的不稳定长回报估计。==  
代码定位：`model/prioritized_nstep_replay_buffer.py:199-221`。

==context==PER 增加 `per_clip_max` 裁剪，抑制非平稳阶段 outlier TD-error 的长期主导效应。==  
代码定位：`model/prioritized_nstep_replay_buffer.py:607-609`。

==context==PER 增加 `per_constraint_priority`（默认关闭），用约束违反度提升约束边界样本被采概率。==  
代码定位：`model/prioritized_nstep_replay_buffer.py:328-358`、`model/prioritized_nstep_replay_buffer.py:597-603`。

==context==PER 增加 mix sampling（默认关闭），引入均匀样本分量缓解 PER 偏置。==  
代码定位：`model/prioritized_nstep_replay_buffer.py:482-497`。

==context==环境 `info` 显式提供 `terminated/truncated/done`，并在失败路径保持一致，便于 replay 层统一边界判断。==  
代码定位：`environment/broker.py:491-497`、`environment/broker.py:546-549`。

## 6. 训练循环中的 sample / TD-error / priority update 定位

1. batch 采样：`model/enhanced_ddpg.py:35`。
2. n-step 折扣构建：`model/enhanced_ddpg.py:42-47`。
3. target 计算：`model/enhanced_ddpg.py:54-57`。
4. TD-error 计算：`model/enhanced_ddpg.py:72`。
5. priority 回写：`model/enhanced_ddpg.py:81-89`。
6. beta 记录：`model/enhanced_ddpg.py:91-92`。
7. replay 内 beta 调度：`model/prioritized_nstep_replay_buffer.py:464-476`。

## 7. 与教科书写法的一致性与差异

1. 一致点：训练目标等价于标准 off-policy n-step 目标（具体表达式见“Step 5：与 DDPG target 的衔接”中的公式块，代码在 `model/enhanced_ddpg.py:56`）。
2. 差异点：本实现在 replay 入库阶段先聚合 $G_t^{(n_t)}$ 并存储 `n_steps`，训练时不再回溯原始序列（`model/prioritized_nstep_replay_buffer.py:281-321`）。
3. 差异点：PER 采用可选混采与约束调制，非纯教科书 PER；且均匀混采子集权重固定 1（`model/prioritized_nstep_replay_buffer.py:517-533`）。

## 8. 定位清单

### 8.1 N-step

1. 类与参数：`model/prioritized_nstep_replay_buffer.py:96-180`。
2. done 统一：`model/prioritized_nstep_replay_buffer.py:223-241`。
3. n-step 聚合核心：`model/prioritized_nstep_replay_buffer.py:281-327`。
4. 不连续 step 强制 flush：`model/prioritized_nstep_replay_buffer.py:395-405`。
5. 自适应 n-step：`model/prioritized_nstep_replay_buffer.py:199-221`。

### 8.2 PER

1. sum/min tree：`model/prioritized_nstep_replay_buffer.py:34-84`。
2. PER 抽样：`model/prioritized_nstep_replay_buffer.py:426-439`、`471-543`。
3. IS 权重：`model/prioritized_nstep_replay_buffer.py:520-533`。
4. priority 更新：`model/prioritized_nstep_replay_buffer.py:574-617`。
5. beta 调度：`model/prioritized_nstep_replay_buffer.py:464-469`。

### 8.3 训练循环

1. sample：`model/enhanced_ddpg.py:35`。
2. TD-error：`model/enhanced_ddpg.py:72`。
3. update_priority：`model/enhanced_ddpg.py:81-89`。

### 8.4 done/terminated/truncated 与重启边界

1. 环境 step 定义：`environment/broker.py:483-497`。
2. 失败路径定义：`environment/broker.py:540-549`。
3. 包装层统一 done：`tuner/train.py:751-757`。
4. broker 重启与等待语义：`environment/broker.py:281-357`。
