# BrokerTune Training Update 的 Mosquitto 适配改造说明

## A) Training Update 工作流程（按代码）

### Step 0：从 replay/PER 采样 batch

1. 处理对象：批量样本 $\mathcal{B}=\{(s_i,a_i,r_i,s'_i,d_i)\}_{i=1}^{B}$，其中 $s_i \in \mathbb{R}^{d}$，$a_i \in \mathbb{R}^{m}$，$d_i \in \{0,1\}$；扩展字段含 $w_i^{(IS)} \in \mathbb{R}_{+}$、$\mathrm{idx}_i$、$n_i$、$c_i$。
2. 达到的效果：为每次训练更新提供 TD 目标构造所需字段，并保留 PER priority 回写索引。
3. 数学表达式：

```math
\mathcal{B} = \{(s_i,a_i,r_i,s'_i,d_i,\mathrm{idx}_i,w_i^{(IS)},n_i,c_i)\}_{i=1}^{B}
```

代码定位：`model/enhanced_ddpg.py:76`，`model/prioritized_nstep_replay_buffer.py:562-572`。

### Step 1：计算 target action（含 smoothing 与 clip）

1. 处理对象：下一状态 $s'_i$、目标 actor $\pi_{\phi^-}$、噪声 $\epsilon_i$。
2. 达到的效果：构造平滑后的 target action，并裁剪到动作合法区间。
3. 数学表达式：

```math
\epsilon_i \sim \mathrm{clip}(\mathcal{N}(0,\sigma_{\text{target}}), -c_{\text{noise}}, c_{\text{noise}})
```

```math
a'_i = \mathrm{clip}(\pi_{\phi^-}(s'_i) + \epsilon_i, -1, 1)
```

代码定位：`model/enhanced_ddpg.py:100-103`。

### Step 2：计算 TD target $y$

1. 处理对象：奖励 $r_i$、终止标记 $d_i$、折扣 $\gamma$、可选 $n_i$、目标 critic 值 $Q_{\theta^-}$。
2. 达到的效果：兼容 1-step 与 N-step target；并可选对 target 值做数值裁剪。
3. 数学表达式：

```math
\gamma_i =
\begin{cases}
\gamma^{n_i}, & \text{若 batch 提供 } n_i \\
\gamma, & \text{否则}
\end{cases}
```

```math
y_i = r_i + (1-d_i)\gamma_i\,Q_{\theta^-}(s'_i,a'_i)
```

```math
y_i \leftarrow \mathrm{clip}(y_i,-q_{\text{clip}},q_{\text{clip}}) \quad (q_{\text{clip}}>0)
```

代码定位：`model/enhanced_ddpg.py:92-97`，`model/enhanced_ddpg.py:104-108`。

### Step 3：critic 前向与 loss 计算（IS/约束加权）

1. 处理对象：当前 critic 输出 $Q_{\theta}(s_i,a_i)$、target $y_i$、IS 权重 $w_i^{(IS)}$、约束比率 $c_i$。
2. 达到的效果：支持 `mse` 与 `huber` 两类 critic 损失，并支持约束感知样本加权。
3. 数学表达式：

```math
w_i^{(c)} = 1 + \max(0,c_i),\quad
w_i = w_i^{(IS)}\cdot w_i^{(c)}
```

```math
\mathcal{L}_{\text{critic}}^{\text{mse}} = \frac{1}{B}\sum_{i=1}^{B} w_i\left(Q_{\theta}(s_i,a_i)-y_i\right)^2
```

```math
\mathcal{L}_{\text{critic}}^{\text{huber}} = \frac{1}{B}\sum_{i=1}^{B} w_i\,\mathrm{Huber}\left(Q_{\theta}(s_i,a_i)-y_i\right)
```

代码定位：`model/enhanced_ddpg.py:78-90`，`model/enhanced_ddpg.py:112-118`。

### Step 4：critic 反向传播与优化器 step

1. 处理对象：$\nabla_{\theta}\mathcal{L}_{\text{critic}}$ 与 critic 参数 $\theta$。
2. 达到的效果：完成 critic 参数更新；可选执行梯度范数裁剪抑制异常梯度。
3. 数学表达式：

```math
\theta \leftarrow \theta - \eta_\theta\,\nabla_{\theta}\mathcal{L}_{\text{critic}}
```

```math
g_\theta \leftarrow \mathrm{clip\_norm}(g_\theta, g_{\max}) \quad (g_{\max}>0)
```

代码定位：`model/enhanced_ddpg.py:142-147`。

### Step 5：actor 更新（含延迟更新）

1. 处理对象：actor 参数 $\phi$、critic $Q_{\theta}$、状态 $s_i$。
2. 达到的效果：critic 每步更新；actor 按 `policy_delay` 延迟更新，降低 actor 追噪声风险。
3. 数学表达式：

```math
\mathcal{L}_{\text{actor}} = -\frac{1}{B}\sum_{i=1}^{B} Q_{\theta}(s_i,\pi_{\phi}(s_i))
```

```math
\phi \leftarrow \phi - \eta_\phi\,\nabla_{\phi}\mathcal{L}_{\text{actor}},\quad \text{当 } u \bmod D_{\pi}=0
```

代码定位：`model/enhanced_ddpg.py:171-180`。

### Step 6：soft update（Polyak/tau）

1. 处理对象：在线网络参数与目标网络参数（actor/critic 及 BN 统计）。
2. 达到的效果：延迟更新时同步执行 Polyak 软更新，平滑 target 网络变化。
3. 数学表达式：

```math
\psi^- \leftarrow \tau\psi + (1-\tau)\psi^-
```

代码定位：`model/enhanced_ddpg.py:182-185`。

### Step 7：日志记录与可观测性输出

1. 处理对象：训练更新中的损失、TD 误差、Q 统计、动作统计、梯度范数、配置生效状态。
2. 达到的效果：为噪声/非平稳场景提供训练可观测性，支持快速定位数值不稳定与动作饱和。
3. 数学表达式：

```math
\mathrm{TD\_stats} = \{\mathbb{E}[|\delta|],\ \mathrm{P95}(|\delta|),\ \max(|\delta|)\}
```

```math
\mathrm{Q\_stats} = \{\mathbb{E}[Q],\ \max(Q)\},\quad
\mathrm{Action\_stats} = \{\mathbb{E}[a],\ \min(a),\ \max(a)\}
```

代码定位：`model/enhanced_ddpg.py:187-235`。

## B) Mosquitto 适配改动点（全部使用 ==context== 高亮）

### 改动 1：提高样本利用率（UTD ratio）

- ==context==新增 `utd_ratio`，把每次训练调用的更新次数从 `gradient_steps` 扩展为 `effective_gradient_steps=gradient_steps*utd_ratio`，用于降低昂贵环境交互下的样本浪费。==
- 代码定位：`model/enhanced_ddpg.py:46-49`，`tuner/train.py:141-146`，`tuner/utils.py:53,198`。
- 默认参数：`--utd_ratio=1`，保持旧行为。
- 影响范围：仅训练更新循环，不改环境交互逻辑。

### 改动 2：延迟策略更新（policy delay）

- ==context==新增 `policy_delay`，critic 每次更新，actor 与 target 网络每 `policy_delay` 次 critic 更新才执行，降低非平稳/噪声下 actor 追噪声的风险。==
- 代码定位：`model/enhanced_ddpg.py:24,171-185`，`tuner/train.py:148-153`，`tuner/utils.py:54,199`。
- 默认参数：`--policy_delay=1`，等价旧行为。
- 影响范围：仅训练更新频率策略，不改环境交互。

### 改动 3：critic 损失鲁棒化（Huber + 梯度裁剪）

- ==context==新增 `critic_loss` 支持 `mse/huber`，在 outlier TD target 出现时可切换 Huber 损失提升鲁棒性。==
- ==context==新增 `grad_clip_norm`，当阈值大于 0 时对 actor/critic 梯度范数裁剪，抑制异常梯度导致的参数爆炸。==
- 代码定位：`model/enhanced_ddpg.py:25,35-37,112-118,144-146,177-179`，`tuner/train.py:155-168`，`tuner/utils.py:55-57,200-202`。
- 默认参数：`--critic_loss=mse`、`--grad_clip_norm=0.0`，保持旧行为。
- 影响范围：仅训练更新的 loss 与优化步骤。

### 改动 4：target 值数值稳定保护

- ==context==新增 `target_q_clip`，代码实际做法是对 `target_q_values` 直接 clamp 到 $[-q_{clip},q_{clip}]$，用于缓解非平稳与测量噪声下 target 爆炸与 NaN 风险。==
- 代码定位：`model/enhanced_ddpg.py:107-108`，`tuner/train.py:170-175`，`tuner/utils.py:57,202`。
- 默认参数：`--target_q_clip=0.0`（关闭），保持旧行为。
- 影响范围：仅训练更新的 target 构造，不改环境交互。

### 改动 5：约束感知更新权重（可选）

- ==context==新增 `use_constraint_weighting`，仅使用 batch 现有字段 `constraint_ratios` 构造约束权重 $w_i^{(c)}=1+\max(0,c_i)$，并与 PER IS 权重相乘用于 critic loss。==
- ==context==当 batch 不含 `constraint_ratios`（例如非自定义 replay）时自动退化为 $w_i^{(c)}=1$，不破坏训练流程。==
- 代码定位：`model/enhanced_ddpg.py:85-90`，`model/prioritized_nstep_replay_buffer.py:360-372,562-572`，`tuner/train.py:177-183`，`tuner/utils.py:58,203`。
- 默认参数：`--use_constraint_weighting=0`（关闭），保持旧行为。
- 影响范围：仅 critic 样本加权，不改环境交互。

### 改动 6：训练可观测性增强（必须）

- ==context==新增训练日志：`critic_loss`、`actor_loss`、`td_error_mean/p95/max`、`q_value_mean/max`、`action_mean/min/max`、`critic_grad_norm`、`actor_grad_norm`，并记录 `utd_ratio/policy_delay/effective_gradient_steps/constraint_weight_mean/constraint_weighting_active`。==
- 代码定位：`model/enhanced_ddpg.py:187-235`。
- 默认参数：日志增强默认启用；梯度范数日志仅在 `grad_clip_norm>0` 且发生对应更新时输出。
- 影响范围：仅训练更新模块可观测性输出，不改环境交互。

## 代码接口与默认值汇总

1. `--utd_ratio`：默认 `1`。
2. `--policy_delay`：默认 `1`。
3. `--critic_loss`：默认 `mse`。
4. `--grad_clip_norm`：默认 `0.0`。
5. `--target_q_clip`：默认 `0.0`。
6. `--use_constraint_weighting`：默认 `0`（false）。

对应代码入口：`tuner/train.py:140-183` -> `tuner/utils.py:53-58,198-203` -> `model/enhanced_ddpg.py:20-40`。

## 影响边界说明

- ==context==本次改造仅触及 training update 路径（采样后目标构造、损失、优化、target 更新、训练日志）。==
- ==context==未改动环境交互、Broker 重启策略、reward 计算、done/terminated/truncated 语义定义。==

