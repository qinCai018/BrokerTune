# Task Plan: RL/DDPG 审计与修复

## Goal
在不改变 BrokerTuner 总体目标的前提下，完成 DDPG 训练/评估链路审计与关键缺陷修复，使训练评估流程可诊断、可复现、可对比 baseline 与 RL。

## Current Phase
Phase 5

## Phases
### Phase 1: Discovery
- [x] 定位训练/评估/环境/DDPG入口
- [x] 梳理组件依赖关系
- **Status:** complete

### Phase 2: Root Cause Analysis
- [x] 按 RL/DDPG清单核对实现
- [x] 找到影响目标达成的关键问题
- **Status:** complete

### Phase 3: Implementation
- [x] 修复环境 reward/info/异常处理
- [x] 修复工作负载延迟探测实现
- [x] 修复训练参数暴露与可复现性
- [x] 重写评估为 baseline vs RL 对比
- **Status:** complete

### Phase 4: Verification
- [x] 语法检查
- [x] 回归测试（4项）
- [x] 训练最小探测并记录阻塞点
- **Status:** complete

### Phase 5: Delivery
- [x] 汇总问题分级、定位、修复方案
- [x] 形成 MVP 验证步骤与后续建议
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Reward 改为吞吐+时延组合 | 与目标“吞吐↑、时延↓”一致，避免只优化吞吐 |
| step() 异常转为失败转移而非崩溃 | 提升训练稳定性，便于诊断 |
| 评估默认输出 baseline vs RL 对比 | 满足可评估性能提升要求 |
| 暴露 replay/noise/seed 等超参 | 提升 DDPG 可审计性和可复现性 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `DDPG.__init__()` 不支持 `replay_buffer_size` | 1 | 改为 SB3 2.4.1 参数名 `buffer_size` |
| 本环境 MQTT 连接报 `Operation not permitted` | 1 | 记录为运行环境权限/网络限制阻塞，代码侧增加失败诊断与回退信息 |
