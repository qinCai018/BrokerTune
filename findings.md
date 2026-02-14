# Findings & Decisions
<!-- 
  WHAT: Your knowledge base for the task. Stores everything you discover and decide.
  WHY: Context windows are limited. This file is your "external memory" - persistent and unlimited.
  WHEN: Update after ANY discovery, especially after 2 view/browser/search operations (2-Action Rule).
-->

## Requirements
<!-- 
  WHAT: What the user asked for, broken down into specific requirements.
  WHY: Keeps requirements visible so you don't forget what you're building.
  WHEN: Fill this in during Phase 1 (Requirements & Discovery).
  EXAMPLE:
    - Command-line interface
    - Add tasks
    - List all tasks
    - Delete tasks
    - Python implementation
-->
<!-- Captured from user request -->
-

## Research Findings
<!-- 
  WHAT: Key discoveries from web searches, documentation reading, or exploration.
  WHY: Multimodal content (images, browser results) doesn't persist. Write it down immediately.
  WHEN: After EVERY 2 view/browser/search operations, update this section (2-Action Rule).
  EXAMPLE:
    - Python's argparse module supports subcommands for clean CLI design
    - JSON module handles file persistence easily
    - Standard pattern: python script.py <command> [args]
-->
<!-- Key discoveries during exploration -->
-

## Technical Decisions
<!-- 
  WHAT: Architecture and implementation choices you've made, with reasoning.
  WHY: You'll forget why you chose a technology or approach. This table preserves that knowledge.
  WHEN: Update whenever you make a significant technical choice.
  EXAMPLE:
    | Use JSON for storage | Simple, human-readable, built-in Python support |
    | argparse with subcommands | Clean CLI: python todo.py add "task" |
-->
<!-- Decisions made with rationale -->
| Decision | Rationale |
|----------|-----------|
|          |           |

## Issues Encountered
<!-- 
  WHAT: Problems you ran into and how you solved them.
  WHY: Similar to errors in task_plan.md, but focused on broader issues (not just code errors).
  WHEN: Document when you encounter blockers or unexpected challenges.
  EXAMPLE:
    | Empty file causes JSONDecodeError | Added explicit empty file check before json.load() |
-->
<!-- Errors and how they were resolved -->
| Issue | Resolution |
|-------|------------|
|       |            |

## Resources
<!-- 
  WHAT: URLs, file paths, API references, documentation links you've found useful.
  WHY: Easy reference for later. Don't lose important links in context.
  WHEN: Add as you discover useful resources.
  EXAMPLE:
    - Python argparse docs: https://docs.python.org/3/library/argparse.html
    - Project structure: src/main.py, src/utils.py
-->
<!-- URLs, file paths, API references -->
-

## Visual/Browser Findings
<!-- 
  WHAT: Information you learned from viewing images, PDFs, or browser results.
  WHY: CRITICAL - Visual/multimodal content doesn't persist in context. Must be captured as text.
  WHEN: IMMEDIATELY after viewing images or browser results. Don't wait!
  EXAMPLE:
    - Screenshot shows login form has email and password fields
    - Browser shows API returns JSON with "status" and "data" keys
-->
<!-- CRITICAL: Update after every 2 view/browser operations -->
<!-- Multimodal content must be captured as text immediately -->
-

---
<!-- 
  REMINDER: The 2-Action Rule
  After every 2 view/browser/search operations, you MUST update this file.
  This prevents visual information from being lost when context resets.
-->
*Update this file after every 2 view/browser/search operations*
*This prevents visual information from being lost*

## 2026-02-03
- Initialized planning files in BrokerTuner root.
- Project root contains environment/, model/, tuner/, script/, docs/, requirements.txt, README.md, START_TRAINING.md.
- `environment/broker.py` has reward computation around `_compute_reward` (~line 576) with current config using reward_scale/weight_base/weight_step and tanh/clipping; step() logs reward and clips to +/-1e6.
- `MosquittoBrokerEnv` step uses normalized throughput/latency from state indices (1,5); reward currently throughput-based with delta_base/step, tanh or clip, and reward_scale/weight_base/weight_step. `_sample_state` currently uses placeholder latency_p50=10ms/latency_p95=50ms and queue_depth=0.0 in `build_state_vector`.
- `environment/config.py` EnvConfig includes baseline/step intervals, state_dim=10, action_dim=11, reward params (scale/weights/clip/tanh). No latency probe or replay buffer params yet.
- `environment/utils.py` already uses `paho-mqtt` in `MQTTSampler` with `_ensure_mqtt_available`; latency probe will likely extend here or in workload manager. `build_state_vector` defined around line 295.
- `script/workload.py` defines `WorkloadManager` using emqtt_bench with start/stop/restart and process management; no latency probe support currently.
- `model/ddpg.py` defines CustomActor/Critic and CustomDDPGPolicy; current architecture already uses Linear/LeakyReLU/BN/Tanh/Dropout/Sigmoid pattern but `tuner/utils.py` still instantiates DDPG with `policy="MlpPolicy"` and no custom replay buffer.
- `tuner/train.py` defines CLI defaults (tau=0.005, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, batch_size=128) and includes `ActionThroughputLoggerWrapper` for CSV logging around line ~368.
- `ActionThroughputLoggerWrapper` currently logs CSV header with throughput+reward only; row includes action list + decoded knobs + sys metrics + throughput + reward. No latency or reward components columns.
- `tuner/train.py` uses `make_ddpg_model` after `record_default_baseline`; no replay buffer params passed; env config currently not extended. CLI doesn't include replay buffer params.
- `requirements.txt` already includes `paho-mqtt` alongside `stable-baselines3[extra]` and `gymnasium`.
- Attempted to read `/home/qincai/userDir/Tuner/CDBTuner/model/ddpg.py` but file not found; need to locate CDBTuner model path.
- CDBTuner structure: `CDBTuner/models` exists; `rg` shows ddpg import in `models/__init__.py` (need to locate actual DDPG implementation file under `CDBTuner/models`).
- CDBTuner `models/ddpg.py` actor/critic architecture matches current BrokerTuner CustomActor/Critic (Linear/LeakyReLU/BN/Tanh/Dropout, Sigmoid output; critic uses separate state/action nets then concat and similar layers).
- Local environment lacks `stable_baselines3` import (ImportError), so SB3 source not inspectable here.
- No dedicated test suite found (only `misc/verify/throughput_test.py` and docs mention pytest).
- User selected approach: implement SB3 DDPG subclass to update PER priorities using TD-error while keeping SB3 training engine.

## 2026-02-13
- 训练/评估主入口确认：`tuner/train.py`、`tuner/evaluate.py`。
- 环境主实现确认：`environment/broker.py`（`reset/step/_sample_state/_compute_reward`）。
- 发现并修复关键偏差：
  - 奖励函数仅用吞吐量，未显式优化时延；
  - 时延数据是硬编码常量（10ms/50ms）；
  - `info` 诊断字段不足，难以做性能归因；
  - 评估脚本仅输出 reward，无 baseline 对比吞吐/时延；
  - DDPG 超参（buffer、learning_starts、train_freq、noise、seed）缺少显式配置入口。
- 新增延迟探测：`WorkloadManager` 中实现 MQTT 回环 probe，支持 `p50/p95/min/max/samples`。
- 新增回归测试：
  - `tests/test_env_reward.py`（奖励偏好低时延 + 失败转移截断）
  - `tests/test_evaluate_metrics.py`（评估指标提取与汇总）
- 本地验证结论：
  - 语法检查通过；
  - 4 个测试通过；
  - 最小训练探测在当前沙箱环境受 MQTT 连接权限限制（`Operation not permitted`），属于运行环境限制而非代码语法错误。
