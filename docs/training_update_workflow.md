# BrokerTune 训练更新（Training Update）工作流程与 Mosquitto 适配说明

## 1. 真实实现定位与调用链（按代码）

1. 训练入口与主循环。
- `tuner/train.py:1281-1700`：`main()` 创建环境与模型，并调用 `model.learn(...)`。
- `tuner/train.py:1517-1555`：`make_ddpg_model(...)` 参数透传。
- `tuner/train.py:1696-1700`：训练主循环触发点 `model.learn(total_timesteps=..., callback=...)`。

2. training update 真实实现。
- `model/enhanced_ddpg.py:42-235`：`EnhancedDDPG.train()` 覆盖父类训练更新逻辑。

3. replay/PER 采样与返回字段。
- `model/enhanced_ddpg.py:76`：`replay_data = self.replay_buffer.sample(...)`。
- `model/prioritized_nstep_replay_buffer.py:562-572`：返回 `observations/actions/next_observations/dones/rewards/indices/weights/n_steps/constraint_ratios`。

4. target 计算（含 target action noise/clip/smoothing）。
- `model/enhanced_ddpg.py:100-103`：target action 构造，带高斯噪声与 clip。
- `model/enhanced_ddpg.py:104-108`：target critic 最小化与 `target_q_values` 构造（可选 target clamp）。

5. critic loss 与反向传播。
- `model/enhanced_ddpg.py:112-118`：`mse/huber` critic loss（样本权重加权）。
- `model/enhanced_ddpg.py:142-147`：`critic_loss.backward()` 与 `critic.optimizer.step()`（可选 grad clipping）。

6. actor update。
- `model/enhanced_ddpg.py:171-180`：`if self._n_updates % self.policy_delay == 0` 条件下更新 actor（`-Q` 目标）。

7. soft update（Polyak/tau）。
- `model/enhanced_ddpg.py:182-185`：更新 actor/critic target 参数与 BN 统计。

8. done/terminated/truncated 来源与语义（$d \in \{0,1\}$）。
- `environment/broker.py:483-497`：环境正常路径写入 `terminated/truncated/done`。
- `environment/broker.py:540-549`：失败路径同样写入 `terminated/truncated/done`。
- `tuner/train.py:794-801`：wrapper 兼容 4 元组和 5 元组返回。
- `model/prioritized_nstep_replay_buffer.py:236-241`：回放层统一 done 语义（array/info/terminated/truncated 四路 OR）。

9. N-step return 来源确认。
- `model/prioritized_nstep_replay_buffer.py:281-307`：在 replay 缓冲中先计算累计回报并确定实际步长。
- `model/prioritized_nstep_replay_buffer.py:315-320`：将累计回报写入 `rewards`，步长写入 `n_steps`。
- `model/enhanced_ddpg.py:92-97`：训练时读取 `n_steps` 计算 $\gamma^{n}$。
- `model/enhanced_ddpg.py:106`：训练端使用 `replay_data.rewards` 作为回报项拼接 target。

## 2. Training Update 分步流程（Step 0-7）

### Step 0：从 replay/PER 采样 batch

1. 处理对象：$\mathcal{B}=\{(s_i,a_i,r_i,s'_i,d_i,\mathrm{idx}_i,w_i^{(IS)},n_i,c_i)\}_{i=1}^{B}$，其中 $s_i \in \mathbb{R}^{d}$，$a_i \in \mathbb{R}^{m}$，$d_i \in \{0,1\}$，$w_i^{(IS)} \in \mathbb{R}_{+}$，$n_i \in \mathbb{N}_{+}$，$c_i \in \mathbb{R}_{+}$。
2. 达到的效果：一次采样同时得到训练字段、PER 回写索引、IS 权重和 N-step/约束字段。
3. 数学表达式：

```math
\mathcal{B} = \{(s_i,a_i,r_i,s'_i,d_i,\mathrm{idx}_i,w_i^{(IS)},n_i,c_i)\}_{i=1}^{B}
```

代码定位：`model/enhanced_ddpg.py:76`，`model/prioritized_nstep_replay_buffer.py:562-572`。

### Step 1：计算 target action（Target Actor）

1. 处理对象：$s'_i$、target actor 参数 $\theta'$、target policy 噪声 $\epsilon_i$。
2. 达到的效果：通过噪声平滑 target action，并执行动作范围裁剪。
3. 数学表达式：

```math
\epsilon_i \sim \mathrm{clip}(\mathcal{N}(0,\sigma_{\mathrm{target}}),-c_{\mathrm{noise}},c_{\mathrm{noise}})
```

```math
a'_{i} = \mathrm{clip}(\pi_{\theta'}(s'_i)+\epsilon_i,-1,1)
```

代码定位：`model/enhanced_ddpg.py:100-103`。

### Step 2：计算 TD target $y$

1. 处理对象：$r_i$、$d_i$、$n_i$、$Q_{\phi'}$、$a'_i$。
2. 达到的效果：兼容 1-step 与 N-step 的 off-policy target 构造；可选对 target 值做数值裁剪。
3. 数学表达式：

```math
\gamma_i =
\begin{cases}
\gamma^{n_i}, & \text{batch 含 } n_i \\
\gamma, & \text{否则}
\end{cases}
```

```math
y_i = r_i + (1-d_i)\gamma_i Q_{\phi'}(s'_i,a'_i)
```

```math
y_i \leftarrow \mathrm{clip}(y_i,-q_{\mathrm{clip}},q_{\mathrm{clip}}) \quad (q_{\mathrm{clip}}>0)
```

说明：当启用 N-step 时，`replay_data.rewards` 已在入库阶段构造为 $G_t^{(n)}$，训练端再用 $\gamma^{n}$ 乘以 bootstrap 项。

```math
y_t = G_t^{(n_t)} + \gamma^{n_t}(1-d_t)Q_{\phi'}(s_{t+n_t},a'_{t+n_t})
```

代码定位：`model/prioritized_nstep_replay_buffer.py:281-307,315-320`，`model/enhanced_ddpg.py:92-97,106-108`。

### Step 3：Critic 前向与 loss 计算（含 IS weights）

1. 处理对象：$Q_{\phi}(s_i,a_i)$、$y_i$、$w_i^{(IS)}$、可选约束权重 $w_i^{(c)}$。
2. 达到的效果：对 critic 误差做样本级加权；支持 `mse` 与 `huber` 两种损失。
3. 数学表达式：

```math
w_i^{(c)} = 1 + \max(0,c_i), \quad w_i = w_i^{(IS)} \cdot w_i^{(c)}
```

```math
\mathcal{L}_{\mathrm{critic}}^{\mathrm{mse}} = \frac{1}{B}\sum_{i=1}^{B} w_i\left(Q_{\phi}(s_i,a_i)-y_i\right)^2
```

```math
\mathcal{L}_{\mathrm{critic}}^{\mathrm{huber}} = \frac{1}{B}\sum_{i=1}^{B} w_i\,\mathrm{Huber}\left(Q_{\phi}(s_i,a_i)-y_i\right)
```

代码定位：`model/enhanced_ddpg.py:78-90,112-118`。

### Step 4：Critic 反向传播与优化器更新

1. 处理对象：$\nabla_{\phi}\mathcal{L}_{\mathrm{critic}}$ 与 critic 参数 $\phi$。
2. 达到的效果：执行 critic 参数更新；当开启裁剪时约束梯度范数。
3. 数学表达式：

```math
\phi \leftarrow \phi - \eta_{\phi}\nabla_{\phi}\mathcal{L}_{\mathrm{critic}}
```

```math
g_{\phi} \leftarrow \mathrm{clip\_norm}(g_{\phi},g_{\max}) \quad (g_{\max}>0)
```

代码定位：`model/enhanced_ddpg.py:142-147`。

### Step 5：Actor 更新（policy gradient / -Q loss）

1. 处理对象：actor 参数 $\theta$、critic $Q_{\phi}$、状态 $s_i$。
2. 达到的效果：以 `-Q` 目标更新 actor；支持延迟更新频率控制。
3. 数学表达式：

```math
\mathcal{L}_{\mathrm{actor}} = -\frac{1}{B}\sum_{i=1}^{B}Q_{\phi}(s_i,\pi_{\theta}(s_i))
```

```math
\theta \leftarrow \theta - \eta_{\theta}\nabla_{\theta}\mathcal{L}_{\mathrm{actor}}, \quad \text{当 } u \bmod D_{\pi}=0
```

代码定位：`model/enhanced_ddpg.py:171-180`。

### Step 6：Soft update（Target 网络软更新）

1. 处理对象：在线参数 $\theta,\phi$ 与 target 参数 $\theta',\phi'$。
2. 达到的效果：通过 Polyak 软更新平滑 target 网络，提升训练稳定性。
3. 数学表达式：

```math
\theta' \leftarrow \tau\theta + (1-\tau)\theta'
```

```math
\phi' \leftarrow \tau\phi + (1-\tau)\phi'
```

代码定位：`model/enhanced_ddpg.py:182-185`。

### Step 7：与 PER 的 priority 回写衔接

1. 处理对象：TD 误差 $\delta_i$、样本索引 $\mathrm{idx}_i$、可选约束信号 $c_i$。
2. 达到的效果：将训练中估计的样本难度回写到 replay priority，影响后续采样分布。
3. 数学表达式：

```math
\delta_i = \left|Q_{\phi}(s_i,a_i)-y_i\right|
```

```math
u_i = (|\delta_i|+\epsilon)\cdot w_i^{(\mathrm{constraint})}
```

```math
p_i = u_i^{\alpha}, \quad \text{并调用 }\mathrm{update\_priorities}(\mathrm{idx}_i,p_i)
```

代码定位：`model/enhanced_ddpg.py:127-129,149-167`，`model/prioritized_nstep_replay_buffer.py:574-617`。

## 3. done/terminated/truncated 语义确认

1. 处理对象：$d_t \in \{0,1\}$、`terminated`、`truncated`、`info.done`。
2. 达到的效果：环境、wrapper、replay 三层统一终止语义，避免跨 episode 混淆。
3. 数学表达式：

```math
d_t = \mathbf{1}[d_t^{(array)} \lor d_t^{(info)} \lor terminated_t \lor truncated_t]
```

代码定位：`environment/broker.py:483-497,540-549`，`tuner/train.py:794-801`，`model/prioritized_nstep_replay_buffer.py:236-241`。

## 4. Mosquitto 调优适配改动点（高亮）

### 4.1 UTD ratio

==context==新增 `--utd_ratio`，训练端按 `effective_gradient_steps = gradient_steps * utd_ratio` 扩展每次训练调用的更新次数，在环境交互昂贵时提高单样本利用率。==

代码定位：`tuner/train.py:141-146,1528-1530`，`tuner/utils.py:53,198`，`model/enhanced_ddpg.py:46-49`。

默认值与兼容性：`--utd_ratio=1`，默认行为与旧版本一致。

影响范围：仅 training update 更新次数，不改环境交互。

为何更适合 Mosquitto：交互代价高（可能伴随 broker 重启与稳定等待），增加离线更新比可提升吞吐/时延目标学习效率。

### 4.2 Policy delay

==context==新增 `--policy_delay`，critic 每次更新，actor 每 `policy_delay` 次 critic 更新才更新一次，并在同频率执行 target soft update，降低噪声下 actor 过拟合瞬时 TD 误差。==

代码定位：`tuner/train.py:148-153,1530-1531`，`tuner/utils.py:54,199`，`model/enhanced_ddpg.py:171-185`。

默认值与兼容性：`--policy_delay=1`，默认行为与旧版本一致。

影响范围：仅 actor/target 更新频率，不改 collector/env。

为何更适合 Mosquitto：非平稳与测量噪声大时，延迟策略更新可先让 critic 稳定拟合，再驱动 actor。

### 4.3 鲁棒 critic loss + 梯度裁剪

==context==新增 `--critic_loss {mse,huber}`，可切换 Huber 损失抑制 outlier TD target 对梯度的放大。==

==context==新增 `--grad_clip_norm`，阈值大于 0 时对 actor/critic 梯度范数裁剪，降低异常样本导致的更新爆炸风险。==

代码定位：`tuner/train.py:155-168,1531-1533`，`tuner/utils.py:55-57,200-202`，`model/enhanced_ddpg.py:112-118,142-147,175-179`。

默认值与兼容性：`--critic_loss=mse`、`--grad_clip_norm=0.0`，默认行为与旧版本一致。

影响范围：仅 loss 计算和优化器 step。

为何更适合 Mosquitto：时延测量与重启扰动容易产生尖峰样本，鲁棒损失和梯度裁剪可显著提升稳定性。

### 4.4 Target 数值稳定保护

==context==新增 `--target_q_clip`，代码实际做法是对 `target_q_values` 直接做 clamp，限制 target 数值范围以降低 NaN/Inf 风险。==

代码定位：`tuner/train.py:170-175,1533`，`tuner/utils.py:57,202`，`model/enhanced_ddpg.py:106-108`。

默认值与兼容性：`--target_q_clip=0.0`（关闭），默认行为与旧版本一致。

影响范围：仅 TD target 数值范围，不改 reward/env。

为何更适合 Mosquitto：非平稳阶段 target 可突然放大，裁剪可增强训练数值稳定性。

### 4.5 约束感知权重

==context==新增 `--use_constraint_weighting`，当 batch 含 `constraint_ratios` 时构造 $w_i^{(c)}=1+\max(0,c_i)$ 并与 IS 权重相乘，让 critic 更关注约束边界/违约样本。==

==context==约束信号来源于现有字段：`reward_components.latency_violation_ms/latency_limit_ms`（或 `unsafe` 回退）在 replay 中提取为 `constraint_ratios`；若字段缺失自动退化为不加权。==

代码定位：`environment/broker.py:507-508,819-830`，`model/prioritized_nstep_replay_buffer.py:328-358,559-572`，`model/enhanced_ddpg.py:85-90`，`tuner/train.py:177-183,1534`。

默认值与兼容性：`--use_constraint_weighting=0`（false），默认行为与旧版本一致。

影响范围：仅 critic 样本权重，不改环境逻辑。

为何更适合 Mosquitto：目标是“吞吐最大化同时满足时延约束”，该机制可提升约束附近样本学习权重而不完全偏置。

### 4.6 训练可观测性增强

==context==新增训练日志：`td_error_mean/p95/max`、`q_value_mean/max`、`action_mean/min/max`、`critic_grad_norm`、`actor_grad_norm`，并补充 `utd_ratio/policy_delay/effective_gradient_steps/constraint_weight_mean/constraint_weighting_active`。==

代码定位：`model/enhanced_ddpg.py:187-235`。

默认值与兼容性：日志默认记录；梯度范数仅在开启裁剪且发生对应更新时记录。

影响范围：仅 logger 输出，不改环境交互。

为何更适合 Mosquitto：可直接观测噪声冲击、动作饱和、Q 值爆炸与梯度异常，便于在昂贵实验中快速诊断。

## 5. 当前默认行为与兼容性结论

1. 处理对象：新增开关参数集合 $\Omega=\{\texttt{utd\_ratio},\texttt{policy\_delay},\texttt{critic\_loss},\texttt{grad\_clip\_norm},\texttt{target\_q\_clip},\texttt{use\_constraint\_weighting}\}$。
2. 达到的效果：全部开关默认值保持旧行为等价，仅在显式开启时改变训练更新细节。
3. 数学表达式：

```math
\Omega_{\mathrm{default}} = \{1,1,\mathrm{mse},0,0,\mathrm{false}\}
```

```math
\text{默认配置下，训练目标形式与原实现保持一致，仅增加可观测性输出。}
```

