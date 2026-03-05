# FeatureWiseAttentionExtractor（按特征维度门控）在 BrokerTune/APN-DDPG 中的工作流程与数学描述

## 1. 标题与范围声明
本文以仓库当前代码实现为唯一依据，说明 `FeatureWiseAttentionExtractor`（按特征维度门控）的参数语义、前向计算、统计接口、训练接入路径与 BrokerTune 10 维状态语义。  
其中“BrokerTune/APN-DDPG 训练栈”在代码中的实现载体为 `EnhancedDDPG`（`MlpPolicy` + 可选 `FeatureWiseAttentionExtractor`）。

## 2. 代码锚点总览（路径+行号）
1. `model/attention_extractor.py:15-35`：`FeatureWiseAttentionExtractor` 类定义与构造参数。  
2. `model/attention_extractor.py:71-95`：attention 权重计算与 `forward` 主流程。  
3. `model/attention_extractor.py:93-94`：`last_gate_raw` / `last_gate` 的保存。  
4. `model/attention_extractor.py:97-132`：`get_gate_stats` 统计输出逻辑。  
5. `tuner/utils.py:139-145,162-177`：`MlpPolicy` 通过 `features_extractor_class` 接入 attention extractor。  
6. `tuner/train.py:159-180,1426-1443`：`--use_attention` 开关定义（默认关闭）与参数传递路径。  
7. `environment/utils.py:295-388`：BrokerTune 10 维状态向量定义、归一化与索引顺序。

> 若未来行号轻微漂移，可用以下唯一片段定位：  
> `self.last_gate_raw = gate_raw.detach()` / `self.last_gate = effective_gate.detach()`（`model/attention_extractor.py`）  
> `policy_kwargs["features_extractor_class"] = FeatureWiseAttentionExtractor`（`tuner/utils.py`）  
> `state = np.array([clients_norm, msg_rate_norm, ... latency_avg_norm], dtype=np.float32)`（`environment/utils.py`）

## 3. FeatureWiseAttentionExtractor 定义与参数语义
依据 `model/attention_extractor.py:15-35` 与后续初始化代码：

- 输入要求：观测空间必须是 1D `Box`。设状态为 $s \in \mathbb{R}^{d}$，其中 BrokerTune 中通常 $d=10$。  
- 两层门控网络：
  - 第一层：$W_1 \in \mathbb{R}^{m \times d},\ b_1 \in \mathbb{R}^{m}$
  - 第二层：$W_2 \in \mathbb{R}^{d \times m},\ b_2 \in \mathbb{R}^{d}$
  - 其中 $m=\max(4,\texttt{attention\_hidden\_dim})$。
- 可选归一化：`attention_use_layer_norm` 为真时使用 `LayerNorm(d)`，否则 `Identity`。
- 关键稳定化参数：
  - `attention_temperature`：logit 温度缩放（下界 `1e-6`）。
  - `attention_gate_floor`：全局门控下限（裁剪到 $[0,1]$）。
  - `attention_residual_ratio`：残差保留比例（裁剪到 $[0,1]$）。
  - `attention_use_domain_priors`：是否启用 BrokerTune 关键维度先验（且仅当 `features_dim == 10` 生效）。
  - `attention_critical_floor`：关键维度门控下限（裁剪到 $[0,1]$）。
  - `attention_critical_indices=(1,5,6,8,9)`：关键维度索引（越界索引会被过滤）。

## 4. 前向流程 Step 0/1/2/...（代码对齐）
以下流程对应 `model/attention_extractor.py:71-95`（含 `compute_attention_weights` 与 `forward`）。

### Step 0：符号与输入定义
- 处理对象：$s \in \mathbb{R}^{d}$（单样本）或 $S \in \mathbb{R}^{B\times d}$（批量），参数矩阵 $W_1,W_2$。  
- 达到的效果：明确门控作用于“特征维度”，不是时序注意力。  
- 数学表达式：
```math
S = [s^{(1)}, \dots, s^{(B)}]^\top
```

### Step 1：输入清洗（NaN/Inf 处理）
- 处理对象：$S$。  
- 达到的效果：将 `NaN/+Inf/-Inf` 置零，防止门控网络数值异常扩散。  
- 数学表达式：
```math
X = \mathrm{sanitize}(S) = \mathrm{nan\_to\_num}(S)
```

### Step 2：可选 LayerNorm（门控网络输入侧）
- 处理对象：$X$ 与归一化模块（`LayerNorm` 或 `Identity`）。  
- 达到的效果：在可选情况下减轻各维尺度差异；未启用时保持原值直通。  
- 数学表达式：
```math
\tilde{X} =
\begin{cases}
\mathrm{LN}(X), & \text{if } \texttt{attention\_use\_layer\_norm}=1 \\
X, & \text{otherwise}
\end{cases}
```

### Step 3：两层 MLP 生成 logits
- 处理对象：$\tilde{X}, W_1,b_1,W_2,b_2$。  
- 达到的效果：将原始状态映射到与特征维同维度的门控 logits。  
- 数学表达式：
```math
H = \mathrm{ReLU}(\tilde{X}W_1^\top + b_1), \qquad Z = HW_2^\top + b_2
```

### Step 4：温度缩放 + Sigmoid 得到原始门控
- 处理对象：$Z$ 与温度 $\tau$（`attention_temperature`）。  
- 达到的效果：控制门控分布“软/硬”程度，并将值域限制在 $(0,1)$。  
- 数学表达式：
```math
G_{\mathrm{raw}} = \sigma\!\left(\frac{Z}{\tau}\right)
```

### Step 5：全局下限与关键维先验下限
- 处理对象：$G_{\mathrm{raw}}$、全局下限 $g_{\min}$、关键集合 $\mathcal{C}$ 与关键下限 $g_{\mathrm{crit}}$。  
- 达到的效果：避免门控过小导致关键信号被“关死”；对关键索引额外保底。  
- 数学表达式：
```math
G_1 = \max\left(G_{\mathrm{raw}},\ g_{\min}\right)
```
```math
G_2[:, i] =
\begin{cases}
\max(G_1[:, i],\ g_{\mathrm{crit}}), & i \in \mathcal{C} \text{ 且启用领域先验} \\
G_1[:, i], & \text{otherwise}
\end{cases}
```

### Step 6：残差门控融合
- 处理对象：$G_2$ 与残差比例 $\rho$（`attention_residual_ratio`）。  
- 达到的效果：保留最小直通比例，降低梯度早期波动和过抑制风险。  
- 数学表达式：
```math
G = \rho + (1-\rho)\,G_2
```

### Step 7：逐元素门控输出
- 处理对象：清洗后的状态 $X$ 与有效门控 $G$。  
- 达到的效果：按特征维缩放状态后再送入后续策略/价值网络。  
- 数学表达式：
```math
S' = G \odot X
```

```math
s' = g \odot s
```

代码中使用的是逐元素乘 $\odot$，不是 $\oplus$。

### Step 8：保存解释性缓存
- 处理对象：$G_{\mathrm{raw}}$ 与 $G$。  
- 达到的效果：将最近一次前向的门控快照（`detach` 后）保存给统计接口使用。  
- 数学表达式：
```math
\texttt{last\_gate\_raw} \leftarrow \mathrm{detach}(G_{\mathrm{raw}}), \qquad
\texttt{last\_gate} \leftarrow \mathrm{detach}(G)
```

## 5. LayerNorm/归一化分支的真实实现与默认行为
代码存在 LN 分支（`model/attention_extractor.py:48,73`）：

- 若 `attention_use_layer_norm=True`，门控网络输入使用 LN：
```math
\tilde{s} = \mathrm{LN}(s)
```
- 若为默认值 `False`，则为恒等映射：
```math
\tilde{s} = s
```

与常见写法差异（以代码为准）：
- 当前实现的 LN 是“门控网络输入侧归一化”，不是“attention 输出后再 LN”。  
- `tuner/train.py:175-180` 的参数 help 文案写“Attention输出后是否使用LayerNorm”，但实际生效位置由 `model/attention_extractor.py:73` 决定，即 `x_norm = self.layer_norm(x)`。

## 6. 可解释性接口：last_gate_raw / last_gate 与 get_gate_stats

### 6.1 缓存保存方式（代码锚点）
`model/attention_extractor.py:93-94`：

- `self.last_gate_raw = gate_raw.detach()`  
- `self.last_gate = effective_gate.detach()`

含义：保存的是“最近一次前向”门控张量的无梯度副本，不参与反向传播。

### 6.2 get_gate_stats 内置输出字段（代码锚点：97-132）
定义 $G=\texttt{last\_gate} \in \mathbb{R}^{B\times d}$，$R=\texttt{last\_gate\_raw} \in \mathbb{R}^{B\times d}$。  
函数返回（当数据可用时）如下统计量：

- `gate_mean`：
```math
\mu_G = \frac{1}{Bd}\sum_{b=1}^{B}\sum_{i=1}^{d} G_{b,i}
```
- `gate_std`：
```math
\sigma_G = \sqrt{\frac{1}{Bd}\sum_{b=1}^{B}\sum_{i=1}^{d}(G_{b,i}-\mu_G)^2}
```
- `gate_min`：
```math
G_{\min} = \min_{b,i} G_{b,i}
```
- `gate_max`：
```math
G_{\max} = \max_{b,i} G_{b,i}
```
- `gate_raw_mean/std/min/max`：与上式同构，只是将 $G$ 换为 $R$。  
- `gate_critical_mean`（当关键索引集合 $\mathcal{C}$ 非空）：
```math
\mu_{G,\mathcal{C}} = \frac{1}{B|\mathcal{C}|}\sum_{b=1}^{B}\sum_{i\in\mathcal{C}} G_{b,i}
```

当 `last_gate is None` 时，函数返回 `gate_*` 四项为 `0.0`；若 `last_gate_raw` 存在则补充 `gate_raw_*` 四项。

### 6.3 论文/实验常用“派生指标”（非 get_gate_stats 内置字段）
以下可由 `last_gate` 额外计算：

- 门控稀疏度（阈值 $\epsilon$）：
```math
\mathrm{Sparsity}_{\epsilon}(G)=\frac{1}{Bd}\sum_{b=1}^{B}\sum_{i=1}^{d}\mathbf{1}[G_{b,i}\le \epsilon]
```
- 平均门控向量与 top-k 维度：
```math
\bar{g}_i=\frac{1}{B}\sum_{b=1}^{B}G_{b,i},\qquad
\mathrm{TopK}(\bar{g},k)=\operatorname*{arg\,topk}_{i\in\{1,\dots,d\}} \bar{g}_i
```

用途建议：
- `gate_mean/std`：监控门控是否塌缩到常数。  
- `gate_min/max`：检查是否触发 floor 或出现异常饱和。  
- `gate_critical_mean`：验证关键维保底策略是否稳定生效。  
- `Sparsity_\epsilon` 与 top-k：用于论文中的可解释性补充图表（非内置返回字段，需外部计算）。

### 6.4 训练/评估日志记录示例（示例代码，不改主流程）
```python
# 假设 model 已创建（EnhancedDDPG）
extractor = getattr(model.policy.actor, "features_extractor", None)
if extractor is not None and hasattr(extractor, "get_gate_stats"):
    stats = extractor.get_gate_stats()

    # 记录到 SB3 logger
    for k, v in stats.items():
        model.logger.record(f"attention/{k}", float(v))

    # 如需 CSV，可追加写入
    # with open("gate_stats.csv", "a", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=sorted(stats.keys()))
    #     if f.tell() == 0:
    #         writer.writeheader()
    #     writer.writerow({kk: float(vv) for kk, vv in stats.items()})
```

## 7. 训练接入路径：--use_attention -> make_ddpg_model -> MlpPolicy

### 7.1 CLI 开关与默认值
`tuner/train.py:159-180`：
- `--use_attention`（`--use-attention`）取值 `{0,1}`，默认 `0`（关闭）。  
- `--attention_hidden_dim` 默认 `64`。  
- `--attention_use_layer_norm`（`--attention-use-layer-norm`）默认 `0`。

### 7.2 参数传递链路
`tuner/train.py:1426-1443`：
- `model = make_ddpg_model(..., use_attention=bool(args.use_attention), attention_hidden_dim=args.attention_hidden_dim, attention_use_layer_norm=bool(args.attention_use_layer_norm), ...)`

`tuner/utils.py:139-145`：
- 仅当 `use_attention=True` 时，注入：
  - `policy_kwargs["features_extractor_class"] = FeatureWiseAttentionExtractor`
  - `policy_kwargs["features_extractor_kwargs"] = {"attention_hidden_dim":..., "attention_use_layer_norm":...}`

`tuner/utils.py:162-177`：
- `EnhancedDDPG(policy="MlpPolicy", ..., policy_kwargs=policy_kwargs if policy_kwargs else None, ...)`

即：
- `use_attention=0` 时，`policy_kwargs=None`，走 SB3 默认特征提取器。  
- `use_attention=1` 时，`MlpPolicy` 的特征提取器替换为 `FeatureWiseAttentionExtractor`。

### 7.3 数据流向与数学抽象
```math
s_t \xrightarrow[]{\mathrm{FWA}} s'_t \xrightarrow[]{\pi} a_t,
\qquad
(s'_t, a_t) \xrightarrow[]{Q} Q(s'_t, a_t)
```

```math
\mathrm{FWA}(s_t)=g_t\odot s_t
```

## 8. BrokerTune 10 维状态语义表（索引、单位/量纲、来源）
依据 `environment/utils.py:310-320,377-388` 与同段归一化逻辑：

| 索引 | 符号 | 含义 | 单位/量纲 | 代码来源与构造顺序 |
|---|---|---|---|---|
| 0 | $s_0$ | 连接数归一化 `clients_norm` | 无量纲（连接数/1000） | `clients_connected = broker_metrics["$SYS/broker/clients/connected"]`，再 `/1000`，写入 `state[0]` |
| 1 | $s_1$ | 消息速率归一化 `msg_rate_norm` | 无量纲（msg/s 再 /10000） | 优先 `$SYS/broker/messages/received_rate`，次选 `$SYS/broker/load/messages/received/1min_per_sec`，受 `uptime` 与 `rate_1min_window_sec` 条件控制后写入 `state[1]` |
| 2 | $s_2$ | CPU 占比 `cpu_ratio` | 无量纲比值 | 由 `build_state_vector` 参数 `cpu_ratio` 直接写入 `state[2]`（上游来自 `/proc` 采样） |
| 3 | $s_3$ | 内存占比 `mem_ratio` | 无量纲比值 | 由参数 `mem_ratio` 写入 `state[3]` |
| 4 | $s_4$ | 上下文切换占比 `ctxt_ratio` | 无量纲比值 | 由参数 `ctxt_ratio` 写入 `state[4]` |
| 5 | $s_5$ | P50 延迟归一化 `latency_p50_norm` | 无量纲（ms/100） | `latency_p50/100` 后写入 `state[5]` |
| 6 | $s_6$ | P95 延迟归一化 `latency_p95_norm` | 无量纲（ms/100） | `latency_p95/100` 后写入 `state[6]` |
| 7 | $s_7$ | 队列深度归一化 `queue_depth_norm` | 无量纲（depth/1000） | `queue_depth/1000` 后写入 `state[7]` |
| 8 | $s_8$ | 最近窗口吞吐均值 `throughput_avg_norm` | 无量纲（与 $s_1$ 同尺度） | `np.mean(throughput_history)`，缺失时回退 `msg_rate_norm`，写入 `state[8]` |
| 9 | $s_9$ | 最近窗口延迟均值 `latency_avg_norm` | 无量纲（与 $s_5$ 同尺度） | `np.mean(latency_history)`，缺失时回退 `latency_p50_norm`，写入 `state[9]` |

状态向量最终还会执行：
```math
s = \mathrm{nan\_to\_num}(s)
```
以替换 NaN/Inf（`environment/utils.py:393-394`）。

## 9. 为什么按特征维度门控对该 10 维状态有意义（基于代码的有根据推断）
1. 吞吐维度的来源存在条件切换：`s_1` 在 derived rate 与 1min rate 间切换，并受 `uptime` 条件影响（`environment/utils.py:327-355`），分布可能随运行阶段变化。按维门控可在阶段变化时自动调节该维影响。  
2. 历史维度 `s_8,s_9` 来自滑动窗口均值（`environment/utils.py:369-375`），本质上是平滑统计；按维门控可在“瞬时指标 vs 历史指标”间动态配权。  
3. 延迟维度 `s_5,s_6,s_9` 与吞吐维度 `s_1,s_8` 在默认关键索引 `(1,5,6,8,9)` 中被显式保底（`model/attention_extractor.py:35,83`），说明实现层面已将其视为高价值信号。  
4. 资源维 `s_2,s_3,s_4` 与队列维 `s_7` 为归一化比值，可能在不同工作负载下出现尺度/波动差异；门控可抑制不稳定维度对策略的瞬时干扰。  
5. 输入和状态均有 `nan_to_num` 清洗（`model/attention_extractor.py:68-69`、`environment/utils.py:393-394`），按维门控与数值清洗组合可提高异常采样下的鲁棒性。

## 10. ==context== BrokerTune/Mosquitto 定制改动高亮汇总
- ==context== 温度缩放：在 `sigmoid` 前执行 `logits / attention_temperature`（`model/attention_extractor.py:75-76`），用于调节门控软硬度。 ==context==  
- ==context== 全局门控下限：`gate = clamp_min(gate_raw, attention_gate_floor)`（`model/attention_extractor.py:89`），避免门控趋近 0。 ==context==  
- ==context== 关键维保底：对 `(1,5,6,8,9)` 执行 `clamp_min(..., attention_critical_floor)`（`model/attention_extractor.py:35,78-84`）。 ==context==  
- ==context== 残差门控：`effective_gate = residual_ratio + (1-residual_ratio)*gate`（`model/attention_extractor.py:91`），保证最小直通比例。 ==context==  
- ==context== 维度条件启用领域先验：`attention_use_domain_priors` 且 `features_dim == 10` 才启用关键维保底（`model/attention_extractor.py:56`）。 ==context==  
- ==context== 增强统计接口：`get_gate_stats` 除全局统计外输出 `gate_critical_mean`（`model/attention_extractor.py:129-132`），便于跟踪关键维门控质量。 ==context==

## 11. 小结（与常见 attention 写法的差异）
1. 本实现是“逐特征 Sigmoid 门控 + 逐元素乘法”，不是 Softmax 注意力分配。  
2. 实现包含 `gate_floor + critical_floor + residual_ratio` 三重稳定化，而非单一 `sigmoid(gate)`。  
3. LN 为可选且默认关闭，且作用在门控网络输入侧，不是输出后归一化。  
4. 训练接入通过 `MlpPolicy(features_extractor_class=FeatureWiseAttentionExtractor)` 动态开关完成，默认关闭。  
5. 10 维状态语义与关键索引先验在代码中一一可追溯，可直接用于审计与论文复现实验说明。
