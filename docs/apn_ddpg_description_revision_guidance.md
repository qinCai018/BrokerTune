# APN-DDPG 论文描述与 BrokerTune 实现对齐修改指导

## 前言
本文件以当前 BrokerTune 仓库源码为准，针对论文中 APN-DDPG / DDPG 训练机制与流程描述给出逐段批注式修改指导。目标是帮助修正文稿表述，使其与项目的真实实现保持一致，而不是直接替换论文全文。需特别说明的是，论文命名层面可统一使用“APN-DDPG”，但在实现层面，当前项目中对应的代码载体是 `EnhancedDDPG`。

## 适用范围
- 适用于修订 BrokerTune 关于 Mosquitto 参数自适应调优、Actor/Critic 定义、Bellman 方程、经验回放、训练更新与工程执行链的论文化描述。
- 适用于需要保持论文命名“APN-DDPG”不变，但又要与当前代码实现严格对齐的场景。
- 若文稿中的说法与源码不一致，应以源码为准，而非以教科书版 DDPG 流程或外部资料为准。

---

## 段1：APN-DDPG 总体描述

### 原文定位
- “本文针对消息代理（Mosquitto）设计了基于 APN-DDPG 进行参数自适应调优系统 BrokerTune……”
- “该系统的 Actor 通过确定性策略梯度……Critic 则通过最小化……TD 误差进行更新……”

### 存在问题
- 当前项目中并不存在独立的 `APN` 类或独立的 `apn_ddpg.py`；若论文暗示 APN-DDPG 是仓库中的单独算法类，会与实现不符。
- 当前训练入口并不是手写的独立 Actor/Critic 训练器，而是 `EnhancedDDPG(policy="MlpPolicy", ...)` 的组合实现。
- 注意力、PER、N-step 在代码中都不是固定默认启用，而是通过配置开关接入；若论文把它们写成当前实现始终启用，需要修正。
- 原文只写“确定性策略梯度 + TD 误差”过于笼统，未体现 BrokerTune 面向 Mosquitto 参数调优的工程执行语境。

### 源码依据
- `tuner/utils.py::make_ddpg_model`：实例化 `EnhancedDDPG(policy="MlpPolicy", ...)`，并按开关挂接 `FeatureWiseAttentionExtractor` 与 `PrioritizedNStepReplayBuffer`。
- `model/enhanced_ddpg.py::EnhancedDDPG`：当前训练更新逻辑的真实实现载体。
- `model/attention_extractor.py::FeatureWiseAttentionExtractor`：注意力模块实现。
- `model/prioritized_nstep_replay_buffer.py::PrioritizedNStepReplayBuffer`：PER + N-step 回放实现。
- `environment/broker.py::MosquittoBrokerEnv`：Mosquitto 调优环境与执行链实现。

### 修改指导
- 保留“BrokerTune 基于 APN-DDPG 进行参数自适应调优”的论文表述。
- 紧接着补一句实现映射：论文中的 APN-DDPG 在项目实现层面由 `EnhancedDDPG` 及其注意力和回放扩展模块承载。
- 明确区分“论文命名”和“代码类名”：前者统一写 APN-DDPG，后者仅在实现说明中写 `EnhancedDDPG`。
- 不要把注意力、PER、N-step 写成“当前实现固定内建且默认启用”；应写成“BrokerTune 在 APN-DDPG 训练栈中支持这些机制，并可按配置接入”。

### 推荐改写要点
- 保留 “BrokerTune 面向 Mosquitto 参数调优” 的总述，但补充 “其实现载体为 `EnhancedDDPG` 与配套环境、注意力及回放模块”。
- 将“Actor / Critic 更新机制”写为“以确定性策略梯度与 TD 误差为核心，并结合目标网络、经验回放及工程化环境执行链”。
- 若正文要突出 APN-DDPG 名称，可加一句：“在实现上，APN-DDPG 通过 `MlpPolicy`、特征级注意力提取器与优先级 N-step 回放模块集成到 BrokerTune 训练栈中。”
- 建议避免出现“项目中实现了独立 APN 子网络类”之类表述。

### 需人工复核
- 无。

---

## 段2：Actor / Critic 定义

### 原文定位
- “APN-DDPG 中有一个参数化的策略函数……”
- “Critic 函数根据状态 $s_t$ 和动作 $a_t$ 指导策略函数的学习……”

### 存在问题
- 当前 `s_t` 不是抽象任意状态，而是 BrokerTune 明确构造的 10 维状态向量。
- 当前 `a_t` 不是抽象任意连续动作，而是 11 维连续动作，对应 11 个 Mosquitto broker 参数。
- 原文未体现 APN-DDPG 中的注意力模块位置；当前实现是“状态先经过特征级门控，再进入 Actor / Critic”。
- 若只写一般式 `a_t=\mu(s_t;\theta^\mu)` 而不补充 BrokerTune 的状态 / 动作语义，会弱化与项目实现的对应关系。

### 源码依据
- `environment/config.py::EnvConfig.state_dim`：`state_dim = 10`。
- `environment/config.py::EnvConfig.action_dim`：`action_dim = 11`。
- `environment/utils.py::build_state_vector`：10 维状态依次为连接数、消息速率、CPU、内存、上下文切换、P50 延迟、P95 延迟、队列深度、最近 5 步平均吞吐量、最近 5 步平均延迟。
- `environment/knobs.py::BrokerKnobSpace.decode_action`：11 维动作映射到 `max_inflight_messages`、`max_inflight_bytes`、`max_queued_messages`、`max_queued_bytes`、`queue_qos0_messages`、`memory_limit`、`persistence`、`autosave_interval`、`set_tcp_nodelay`、`max_packet_size`、`message_size_limit`。
- `model/attention_extractor.py::FeatureWiseAttentionExtractor.forward`：执行 `effective_gate * x` 的特征级门控。
- `tuner/utils.py::make_ddpg_model`：通过 `features_extractor_class=FeatureWiseAttentionExtractor` 接入注意力模块。

### 修改指导
- 一般式 `a_t=\mu(s_t;\theta^\mu)` 和 `Q(s_t,a_t;\theta^Q)` 可以保留，但应在定义后立即补充 BrokerTune 的具体状态和动作语义。
- 建议加入“门控状态”中间变量，例如 $\tilde{s}_t=\mathrm{Gate}(s_t)\odot s_t$，以对应当前的 feature-wise attention。
- 动作定义建议补充一句：该动作对应 11 个 Mosquitto 参数，而不是抽象连续控制量。
- 如果论文不想展开全部 10 维 / 11 维细节，至少要在正文或表格中说明它们分别对应 Broker 状态指标与参数配置项。

### 推荐改写要点
- 在策略函数定义后追加一句：“在 BrokerTune 中，$s_t$ 为 10 维状态向量，$a_t$ 为映射到 11 个 Mosquitto 参数的连续动作。”
- 增补公式：

  ```latex
  \tilde{s}_t = \mathrm{Gate}(s_t)\odot s_t,\qquad
  a_t = \mu(\tilde{s}_t;\theta^\mu)
  ```

- 对 Critic 的定义可改为：“Critic 在门控后的状态 $\tilde{s}_t$ 与动作 $a_t$ 上评估动作价值。”
- 如果论文需要强调 APN 含义，建议把注意力描述为“特征级门控（feature-wise attention）”，不要写成时序 attention。

### 需人工复核
- 无。

---

## 段3：Bellman 方程

### 原文定位
- “$Q^\mu(s_t,a_t;\theta^Q)=\mathbb{E}[r(s_t,a_t)+\gamma Q^\mu(s_{t+1},\mu(s_{t+1};\theta^\mu))]$”

### 存在问题
- 该式只反映了基础 1-step 目标，未体现 BrokerTune 当前实现中与 `n_step`、`target_q_clip` 相关的扩展能力。
- Bellman 目标在项目实现中使用 target actor 与 target critic，而不是直接用在线网络。
- 若论文直接把该式当作当前实现的完整训练目标，会遗漏 N-step 与目标值裁剪等实现细节。

### 源码依据
- `model/enhanced_ddpg.py::EnhancedDDPG.train`：通过 `actor_target` 与 `critic_target` 构造目标值。
- `model/enhanced_ddpg.py::EnhancedDDPG.train`：读取 `replay_data.n_steps`，按 $\gamma^n$ 计算折扣项。
- `model/enhanced_ddpg.py::EnhancedDDPG.train`：当 `target_q_clip > 0` 时对 `target_q_values` 做 clamp。
- `model/prioritized_nstep_replay_buffer.py::PrioritizedNStepReplayBuffer.add`：N-step 累计回报在回放缓冲内部聚合后入库。

### 修改指导
- 可以保留经典 Bellman 形式作为基础定义，但应在后文明确说明 BrokerTune 中实际训练目标由目标网络给出。
- 建议补一句：当启用 N-step 时，目标值中的折扣项写为 $\gamma^{n_t}$，即时回报替换为累计回报 $G_t^{(n_t)}$。
- 建议在 Bellman 方程附近增加一条实现说明：目标值还可按配置执行数值裁剪，以增强训练稳定性。

### 推荐改写要点
- 目标值建议统一写成 target-network 形式：

  ```latex
  y_t = r_t + \gamma Q\bigl(s_{t+1},\mu(s_{t+1};\theta^{\mu'})\mid \theta^{Q'}\bigr)
  ```

- 若要与实现更贴近，建议改写为兼容 N-step 的形式：

  ```latex
  y_t = G_t^{(n_t)} + \gamma^{n_t}(1-d_t)\,
  Q\bigl(s_{t+n_t},\mu(s_{t+n_t};\theta^{\mu'})\mid \theta^{Q'}\bigr)
  ```

- 可在文字中补一句：“当启用 `target_q_clip` 时，$y_t$ 还会被限制在预设区间内。”

### 需人工复核
- 若正文打算写“目标动作平滑噪声”为固定机制，需要结合当前本地 `stable_baselines3==2.4.1` 再核对；该版本 DDPG 默认 `target_noise_clip=0.0`，代码路径存在但默认可退化为零噪声。

---

## 段4：Critic 损失

### 原文定位
- “Critic 的更新可通过最小化 TD 误差对应的损失函数来实现……”
- “$\min L(\theta^Q)=\mathbb{E}[(Q-y_t)^2]$”

### 存在问题
- 当前实现不只支持基础 MSE，还支持 Huber 损失。
- 启用 PER 时，Critic 损失会乘以 IS weights，而不是普通均匀采样的无权重 TD 误差。
- 启用 constraint weighting 时，Critic 损失还会额外乘以约束权重。
- 若论文只保留简单平方误差而完全不说明加权扩展，容易与实现脱节。

### 源码依据
- `model/enhanced_ddpg.py::EnhancedDDPG.train`：支持 `critic_loss in {"mse","huber"}`。
- `model/enhanced_ddpg.py::EnhancedDDPG.train`：`sample_weights = replay_data.weights`，参与 Critic loss 计算。
- `model/enhanced_ddpg.py::EnhancedDDPG.train`：`constraint_weights = 1 + clamp_min(constraint_ratios, 0)`。
- `model/prioritized_nstep_replay_buffer.py::PrioritizedNStepReplayBuffer.sample`：采样返回 `weights` 与 `constraint_ratios`。

### 修改指导
- 若正文需要简洁，可保留基础 MSE 形式作为核心思想。
- 但建议在公式后或段尾补充一句：BrokerTune 实现可使用带 IS 权重和约束权重的加权 TD 损失，并支持 MSE / Huber 两种形式。
- 推荐将原来的“最小化 TD 误差”改成“最小化加权 TD 损失”，更贴近项目实现。

### 推荐改写要点
- 推荐公式样式：

  ```latex
  L_Q = \mathbb{E}_{(s_t,a_t,r_t,s_{t+1})\sim\mathcal{D}}
  \bigl[w_t\,\ell\bigl(Q(s_t,a_t;\theta^Q)-y_t\bigr)\bigr]
  ```

- 可补充说明：其中 $w_t$ 在启用 PER 时包含 IS 权重，在启用约束加权时还可叠加约束权重。
- 若需要兼容原文写法，可写成“当 $w_t=1$ 且 $\ell(x)=x^2$ 时，退化为标准 DDPG 的 MSE 形式”。

### 需人工复核
- 无。

---

## 段5：策略梯度公式

### 原文定位
- “$\nabla_{\theta^\mu}J \approx \mathbb{E}[\nabla_a Q \cdot \nabla_{\theta^\mu}\mu]$”

### 存在问题
- 当前实现中 Actor 的更新在代码层面是通过最小化 `-Q` 目标实现的，而不是直接手写链式法则梯度式。
- Actor 更新并非每次 Critic 更新后都执行，而是受 `policy_delay` 控制。
- 目标网络的 soft update 也与 Actor 更新频率绑定。

### 源码依据
- `model/enhanced_ddpg.py::EnhancedDDPG.train`：`actor_loss = -self.critic.q1_forward(...).mean()`。
- `model/enhanced_ddpg.py::EnhancedDDPG.train`：`if self._n_updates % self.policy_delay == 0` 时才更新 Actor。
- `model/enhanced_ddpg.py::EnhancedDDPG.train`：Actor 更新后执行 `polyak_update` 更新 target 网络。

### 修改指导
- 策略梯度公式本身可以保留，因为它表达了 DPG 的理论依据。
- 但建议在公式后补充一句实现说明：在 BrokerTune 中，Actor 的优化以 `-Q` 目标实现，且按 `policy_delay` 延迟执行。
- 还应补充说明目标网络在相同频率上做 soft update，以与当前实现一致。

### 推荐改写要点
- 保留原始 DPG 公式。
- 增补一句：“在实现层面，Actor 通过最小化 $L_\mu=-\mathbb{E}[Q(s_t,\mu(s_t;\theta^\mu))]$ 更新。”
- 再补一句：“BrokerTune 中策略更新按 `policy_delay` 延迟执行，目标网络在同频率进行 Polyak 软更新。”

### 需人工复核
- 无。

---

## 段6：与 DQN 的比较

### 原文定位
- “其训练机制在形式上与 DQN 相似（经验回放与目标网络），但关键区别在于动作空间为连续且下一时刻动作由目标 Actor 生成。”
- “强化学习中的 DDPG 调优算法使用图……描述。”

### 存在问题
- “与 DQN 相似”这句话若范围不收敛，容易让读者误以为整体训练流程几乎等同于 DQN。
- 当前实现除了连续动作与 target actor 外，还存在注意力、PER、N-step、Broker 环境执行链等差异。
- 若配图标题仍然是“DDPG 网络架构图”，而正文已统一使用 APN-DDPG，命名上会显得不统一。

### 源码依据
- `tuner/utils.py::make_ddpg_model`：训练入口使用 `MlpPolicy`，并按开关接入注意力和回放扩展。
- `model/attention_extractor.py::FeatureWiseAttentionExtractor`：当前实现包含特征级门控。
- `model/prioritized_nstep_replay_buffer.py::PrioritizedNStepReplayBuffer`：当前实现支持优先级 N-step 回放。
- `environment/broker.py::MosquittoBrokerEnv.step`：环境交互涉及 broker 参数应用、重启与多源采样，不是 DQN 常见的简单仿真环境步进。

### 修改指导
- 建议把“与 DQN 相似”限定为“在经验回放与目标网络机制上相似”。
- 随后紧接着写明差异：连续动作空间、target actor 生成下一动作、APN-DDPG 的注意力与回放扩展、以及 BrokerTune 面向 Mosquitto 参数调优的工程执行链。
- 若图注要与正文统一，建议把“DDPG 网络架构图”改成“APN-DDPG / BrokerTune 网络架构图”或类似表述。

### 推荐改写要点
- 建议把原句改成：“APN-DDPG 在目标网络与经验回放机制上与 DQN 具有相似性，但其动作空间连续，下一时刻动作由目标 Actor 生成，并可结合特征级注意力与优先级 N-step 回放机制。”
- 对配图建议改为：“APN-DDPG 网络架构图”或“BrokerTune 中 APN-DDPG 训练架构图”。
- 如需进一步区分，可补一句：“此外，BrokerTune 的环境交互包含 Mosquitto 参数写入、Broker 重启、工作负载恢复及多源指标采集过程。”

### 需人工复核
- 无。

---

## 段7：经验回放与目标网络

### 原文定位
- “Timothy P. Lillicrap……指出 DDPG 中和 DQN 一样使用经验回放缓冲区……”
- “首先从历史经验中抽取 $(s_t,r_t,a_t,s_{t+1})$……”

### 存在问题
- 当前项目不是“固定的普通均匀 replay”实现；启用相关开关后，回放池由 `PrioritizedNStepReplayBuffer` 承担。
- 真实 transition 写入路径不是 Actor 直接写入回放池，而是 SB3 采集器经 `collect_rollouts -> _store_transition -> replay_buffer.add` 写入。
- 当前回放样本字段除 $(s_t,a_t,r_t,s_{t+1})$ 外，还应考虑 `d_t`、`indices`、`weights`、`n_steps`、`constraint_ratios` 等实现字段。
- 若只写“从历史经验中抽取”而不说明优先级采样与 N-step 聚合，会弱化 APN-DDPG 的实现特征。

### 源码依据
- `stable_baselines3/common/off_policy_algorithm.py::collect_rollouts`：环境交互后调用 `_store_transition`。
- `stable_baselines3/common/off_policy_algorithm.py::_store_transition`：最终调用 `replay_buffer.add(...)`。
- `model/prioritized_nstep_replay_buffer.py::PrioritizedNStepReplayBuffer.add`：启用 N-step 时先聚合再入库。
- `model/prioritized_nstep_replay_buffer.py::PrioritizedNStepReplayBuffer.sample`：返回 `observations/actions/next_observations/dones/rewards/indices/weights/n_steps/constraint_ratios`。
- `model/prioritized_nstep_replay_buffer.py::PrioritizedNStepReplayBuffer.update_priorities`：根据 TD 误差回写 priority。

### 修改指导
- 建议把“首先从历史经验中抽取”改成两步表述：先通过环境交互收集样本并写入回放池，再从回放池采样用于训练。
- 建议明确写出样本组至少包含 $(s_t,a_t,r_t,s_{t+1},d_t)$。
- 若篇幅允许，应补充说明：启用 N-step 时，累计回报由回放池内部聚合后再写入；启用 PER 时，采样与回写均基于 priority。
- 可以在正文中将 “经验回放缓冲区” 改为更贴近实现的 “优先级 N-step 回放缓冲区（启用时）”。

### 推荐改写要点
- 推荐将原句改写为：“BrokerTune 中，环境交互样本经采集器写入回放池，再从回放池采样 $(s_t,a_t,r_t,s_{t+1},d_t)$ 用于网络更新。”
- 可补一句：“当启用优先级回放时，样本采样还返回重要性采样权重，并在更新后按 TD 误差回写 priority。”
- 可再补一句：“当启用 N-step 时，即时回报由回放池内部聚合为累计回报后再参与目标值计算。”

### 需人工复核
- 若论文正文计划详细写出 SB3 内部函数名，可确认是否需要保留 `collect_rollouts` / `_store_transition` 这类实现级名称；若正文偏理论，可仅保留“采集器写入回放池”的描述。

---

## 段8：训练过程描述

### 原文定位
- “算法在执行的过程中首先从历史经验中抽取……”
- “然后，通过 $V_t = Q(s_t,a_t)$ 评估此刻的实际价值……利用二者的平方差实现对 Critic 网络中参数的优化。”

### 存在问题
- 当前实现流程并不只是“抽样 -> 计算 $V_{t+1}$ -> 算 $V'_t$ -> 更新 Critic”这么简单。
- 实际训练流程包含 `learning_starts`、`train_freq`、`gradient_steps * utd_ratio`、priority 回写、Actor 延迟更新、target soft update。
- 环境层面还包含 Broker 参数解码、配置写入、Broker 完全重启、workload 恢复与 warmup、`$SYS/#` / `/proc` / latency probe / queue depth 采样、reward 与约束计算。
- reward 在 BrokerTune 中并不是抽象任意 $r(s,a)$，而是吞吐提升 + 时延下降的组合，并可叠加时延约束惩罚。
- 当前环境返回的是 `(next_state, reward, terminated, truncated, info)`，其中 `done = terminated or truncated`。

### 源码依据
- `environment/broker.py::MosquittoBrokerEnv.step`：完整环境执行链。
- `environment/knobs.py::BrokerKnobSpace.decode_action`：11 维动作映射到 Mosquitto 参数。
- `environment/knobs.py::apply_knobs`：写独立配置文件并完全重启 Mosquitto。
- `environment/broker.py::_compute_reward`：吞吐提升 + 时延下降 + 可选时延约束惩罚。
- `model/enhanced_ddpg.py::EnhancedDDPG.train`：`learning_starts` 之后按训练频率触发更新，并执行 `gradient_steps * utd_ratio`。
- `model/enhanced_ddpg.py::EnhancedDDPG.train`：先更新 Critic，再按 `policy_delay` 更新 Actor 和 target。
- `tuner/train.py::main`：实际训练入口与 callback 组装。

### 修改指导
- 建议将训练过程重写为“两层流程”：环境交互层与网络更新层。
- 环境交互层应体现 BrokerTune 的工程特征：动作解码为 broker 参数、配置写入与完全重启、工作负载恢复、指标采样、reward/constraint 计算。
- 网络更新层应体现：从 replay 采样 batch，构造 target，更新 Critic，按条件更新 Actor 与 target，并在启用 PER 时回写 priority。
- 应明确 done 语义：`done = terminated or truncated`。
- 若篇幅允许，应补一句说明训练触发条件：只有在超过 `learning_starts` 且满足 `train_freq` 时才启动参数更新。

### 推荐改写要点
- 建议把原来的单段叙述拆成两句或两段：
  - 第一段写环境交互：动作映射到 Mosquitto 参数，Broker 应用配置后重启并采样状态与奖励。
  - 第二段写网络更新：从回放池采样 batch，构造目标值，更新 Critic，并按延迟频率更新 Actor 与 target 网络。
- 推荐补充如下实现型描述：

  ```latex
  y_t = r_t + \gamma^{n_t}(1-d_t)\,
  Q\bigl(s_{t+n_t},\mu(s_{t+n_t};\theta^{\mu'})\mid\theta^{Q'}\bigr)
  ```

- 推荐明确 reward 语义：“奖励由吞吐提升项和时延下降项共同构成，并可在约束模式下叠加时延违约惩罚。”
- 推荐补一句：“训练更新在满足经验积累与训练频率条件后触发，并在更新后按 TD 误差回写样本优先级。”

### 需人工复核
- 本地 `stable_baselines3==2.4.1` 下，DDPG 缺省为单 critic，且 `target_noise_clip=0.0`；若论文打算写成“双 critic + target smoothing noise 固定启用”，需要先复核实际运行依赖版本。

---

## 统一修改原则
- 论文命名可以统一使用 `APN-DDPG`，但实现说明中必须明确：`APN-DDPG = EnhancedDDPG`。
- 不要把仓库中不存在的独立 `APN` 类、`apn_ddpg.py` 或虚构子网络写入论文描述。
- 注意力、PER、N-step 在源码中通过配置开关接入，不是当前实现的固定默认启用机制；论文若要以 APN-DDPG 形式显式呈现它们，应在算法描述中写明其算法角色，同时在实现说明中注明“按配置接入”。
- BrokerTune 的状态不是抽象任意状态，而是 10 维 Broker 指标与历史统计构成的状态向量。
- BrokerTune 的动作不是抽象任意连续动作，而是映射到 11 个 Mosquitto 参数的连续配置动作。
- 环境不是抽象 `E.step(a)`，而是包含配置写入、Broker 完全重启、工作负载恢复、warmup、多源指标采样和 reward/constraint 计算的工程执行链。
- 经验回放不是 Actor 直接写入，而是经 SB3 采集器写入；启用回放扩展时，`PrioritizedNStepReplayBuffer` 负责优先级采样、N-step 聚合和 priority 回写。
- 训练更新不应退化成教科书版“一次 Critic 更新 + 一次 Actor 更新”；应体现 `learning_starts`、`train_freq`、`gradient_steps * utd_ratio`、priority 回写、`policy_delay`、target soft update 等实现细节。
- 评估阶段应区分 baseline 默认动作与 RL 策略动作，两者均通过环境 `step()` 获取吞吐量、时延与奖励。

## 建议在论文中统一使用的术语
- `APN-DDPG`：论文算法命名。
- `BrokerTune`：Mosquitto 参数自适应调优系统名称。
- `EnhancedDDPG`：APN-DDPG 在当前项目中的代码实现类名，仅用于实现说明，不建议写入论文算法标题与正文。
- `MlpPolicy`：当前训练入口实际使用的策略主体。
- `feature-wise attention` / `特征级注意力门控`：对应 `FeatureWiseAttentionExtractor`，避免写成时序 attention。
- `prioritized N-step replay` / `优先级 N-step 回放`：对应 `PrioritizedNStepReplayBuffer`。
- `target actor / target critic`：目标网络。
- `done = terminated or truncated`：环境终止语义。
- `constraint weighting` / `约束加权`：Critic loss 的可选约束感知加权机制。
- `target_q_clip`：目标值裁剪，属于数值稳定机制，不应与理论主公式混淆。

## 最后检查清单
- 是否已明确写出 `APN-DDPG = EnhancedDDPG`。
- 是否把注意力、PER、N-step 明确标注为“源码支持、按配置接入”。
- 是否避免声称存在独立 `APN` 类或 `apn_ddpg.py`。
- 是否避免把双 critic 写成当前 BrokerTune 的固定实现。
- 是否体现了 `MlpPolicy`、`MosquittoBrokerEnv.step()`、`FeatureWiseAttentionExtractor`、`PrioritizedNStepReplayBuffer` 的真实角色。
- 是否体现了 Broker 参数写入、Broker 完全重启、workload 恢复与指标采样等工程链路。
