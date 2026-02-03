# 文档索引

本文档提供 BrokerTuner 项目所有文档的快速索引。

## 📁 文档目录结构

```
BrokerTuner/
├── README.md                    # 项目主文档（保留在根目录）
├── docs/                        # 文档目录
│   ├── README.md               # 文档目录说明
│   ├── training/               # 训练相关文档（8个文件）
│   ├── troubleshooting/        # 问题修复文档（9个文件）
│   ├── plans/                  # 设计与计划（1个文件）
│   ├── technical/              # 技术说明（1个文件）
│   └── guides/                 # 通用指南（预留）
└── script/
    └── README.md               # 脚本说明（保留在原位置）
```

## 📚 按类别索引

### 训练相关文档 (`docs/training/`)

| 文档 | 说明 |
|------|------|
| **TRAINING_COMMAND.md** | 训练命令指南，包含快速开始和完整命令示例 |
| **TRAINING_DATA_COLLECTION_FLOW.md** | 训练和数据收集流程详解（最详细的流程文档，851行） |
| **TRAINING_RESULTS_GUIDE.md** | 训练结果使用指南 |
| **TRAINING_STABILITY_GUARANTEES.md** | 训练稳定性保证机制说明 |
| **TRAINING_FLOW_CHECK.md** | 训练流程检查文档 |
| **CODE_FLOW_VERIFICATION.md** | 代码流程验证报告（对照文档检查代码实现） |
| **MOSQUITTO_LOG_CONTROL.md** | Mosquitto日志控制与采集 |
| **DISK_SPACE_OPTIMIZATION.md** | 训练磁盘空间优化建议 |

### 问题修复文档 (`docs/troubleshooting/`)

| 文档 | 说明 |
|------|------|
| **THROUGHPUT_ZERO_FIX.md** | 吞吐量为0的问题修复 |
| **THROUGHPUT_ZERO_FINAL_FIX.md** | 吞吐量为0问题的最终修复方案 |
| **CONNECTION_REFUSED_FIX.md** | 连接被拒绝问题修复 |
| **WORKLOAD_ISSUE_FIX.md** | 工作负载问题修复 |
| **PERMISSION_FIX_GUIDE.md** | 权限问题修复指南 |
| **TRAINING_FIXES.md** | 训练相关问题修复汇总 |
| **TRAINING_ANALYSIS_FIX.md** | 训练分析问题修复 |
| **BROKER_CONNECTION_TROUBLESHOOTING.md** | Broker连接故障排除 |
| **BROKER_FIX_SUMMARY.md** | Broker修复总结 |

### 技术说明文档 (`docs/technical/`)

| 文档 | 说明 |
|------|------|
| **ACTION_ADJUSTMENT_MODE.md** | Action调整模式说明（绝对调整 vs 增量调整） |

## 🗺️ 使用场景导航

### 场景1：新手入门
1. **README.md** (根目录) - 了解项目概况
2. **docs/training/TRAINING_COMMAND.md** - 学习如何开始训练
3. **docs/training/TRAINING_DATA_COLLECTION_FLOW.md** - 深入了解训练流程

### 场景2：遇到问题
1. **docs/troubleshooting/** - 根据问题类型查找对应修复文档

### 场景3：代码审查
1. **docs/training/CODE_FLOW_VERIFICATION.md** - 验证代码实现是否符合文档
2. **docs/training/TRAINING_DATA_COLLECTION_FLOW.md** - 查看详细流程说明

### 场景4：训练结果分析
1. **docs/training/TRAINING_RESULTS_GUIDE.md** - 训练结果与日志使用指南

### 场景5：技术深入
1. **docs/technical/ACTION_ADJUSTMENT_MODE.md** - 了解Action调整机制
2. **docs/training/TRAINING_STABILITY_GUARANTEES.md** - 了解稳定性保证

## 📊 文档统计

- **总文档数**: 20个（不含根目录 README.md 与 docs/README.md）
- **训练相关**: 8个
- **问题修复**: 9个
- **技术说明**: 1个
- **设计与计划**: 1个
- **其他**: 1个（script/README.md）

## 🔄 文档更新记录

- **2025-01-11**: 完成文档归类整理，创建文档目录结构

## 📝 注意事项

1. **README.md** 保留在项目根目录，作为项目主文档
2. **script/README.md** 保留在 script 目录，作为脚本说明
3. 所有其他文档已归类到 `docs/` 目录下
4. 新增文档时，请根据内容归类到相应子目录
