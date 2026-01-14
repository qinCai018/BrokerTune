# 文档目录

本目录包含 BrokerTuner 项目的所有文档，按类别组织。

## 目录结构

```
docs/
├── training/          # 训练相关文档
├── troubleshooting/  # 问题修复和故障排除文档
├── analysis/          # 分析报告
├── technical/         # 技术说明文档
└── guides/           # 通用指南（预留）
```

## 文档分类说明

### 📚 training/ - 训练相关文档

包含训练流程、命令、结果分析等文档：

- **TRAINING_COMMAND.md** - 训练命令指南，包含快速开始和完整命令示例
- **TRAINING_DATA_COLLECTION_FLOW.md** - 训练和数据收集流程详解（最详细的流程文档）
- **TRAINING_RESULTS_GUIDE.md** - 训练结果使用指南
- **TRAINING_STABILITY_GUARANTEES.md** - 训练稳定性保证机制说明
- **TRAINING_FLOW_CHECK.md** - 训练流程检查文档
- **CODE_FLOW_VERIFICATION.md** - 代码流程验证报告（对照文档检查代码实现）

### 🔧 troubleshooting/ - 问题修复文档

包含各种问题修复和故障排除文档：

- **THROUGHPUT_ZERO_FIX.md** - 吞吐量为0的问题修复
- **THROUGHPUT_ZERO_FINAL_FIX.md** - 吞吐量为0问题的最终修复方案
- **CONNECTION_REFUSED_FIX.md** - 连接被拒绝问题修复
- **WORKLOAD_ISSUE_FIX.md** - 工作负载问题修复
- **PERMISSION_FIX_GUIDE.md** - 权限问题修复指南
- **TRAINING_FIXES.md** - 训练相关问题修复汇总
- **TRAINING_ANALYSIS_FIX.md** - 训练分析问题修复
- **BROKER_CONNECTION_TROUBLESHOOTING.md** - Broker连接故障排除
- **BROKER_FIX_SUMMARY.md** - Broker修复总结

### 📊 analysis/ - 分析报告

包含各种分析报告：

- **RL_TRAINING_ISSUES_ANALYSIS.md** - 强化学习训练问题分析报告
- **TRAINING_ANALYSIS.md** - 训练分析报告
- **ACTION_THROUGHPUT_LOG.md** - Action和吞吐量日志分析

### 🔬 technical/ - 技术说明文档

包含技术实现细节说明：

- **ACTION_ADJUSTMENT_MODE.md** - Action调整模式说明（绝对调整 vs 增量调整）

## 快速导航

### 新手入门
1. 阅读根目录的 **README.md** 了解项目概况
2. 查看 **training/TRAINING_COMMAND.md** 开始训练
3. 参考 **training/TRAINING_DATA_COLLECTION_FLOW.md** 了解详细流程

### 遇到问题
1. 查看 **troubleshooting/** 目录下的相关修复文档
2. 参考 **analysis/** 目录下的分析报告了解问题原因

### 深入了解
1. 阅读 **training/CODE_FLOW_VERIFICATION.md** 了解代码实现
2. 查看 **technical/** 目录了解技术细节
3. 参考 **analysis/** 目录了解性能分析

## 文档维护

- 新增文档时，请根据内容归类到相应目录
- 重要更新请在文档顶部添加更新日期和版本信息
- 保持文档结构清晰，便于查找
