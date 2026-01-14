#!/bin/bash
# 修复checkpoints目录的文件权限

echo "修复 checkpoints 目录的文件权限..."

# 检查是否有sudo权限
if [ "$EUID" -eq 0 ]; then
    # 已经是root用户
    chown -R qincai:qincai /home/qincai/userDir/BrokerTuner/checkpoints/
    chmod -R u+rw /home/qincai/userDir/BrokerTuner/checkpoints/
    echo "✅ 权限已修复"
else
    # 需要sudo权限
    echo "需要sudo权限来修改文件所有者..."
    sudo chown -R qincai:qincai /home/qincai/userDir/BrokerTuner/checkpoints/
    sudo chmod -R u+rw /home/qincai/userDir/BrokerTuner/checkpoints/
    echo "✅ 权限已修复"
fi

echo ""
echo "当前文件权限："
ls -la /home/qincai/userDir/BrokerTuner/checkpoints/ | head -10
