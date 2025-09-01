# Mujoco_Robot
Reinforcement learning training agent

📖 项目概述
本项目用于学习和验证PPO算法，训练了MuJoCo的Humanoid和Pusher两个环境。

支持的MuJoCo环境
Humanoid-v5 - 训练类人机器人行走
Pusher-v5 - 训练机械臂

🎯 训练结果
奖励函数曲线

策略网络测试结果

🧠 项目设计思路
环境一：Humanoid-v5
状态空间：一维向量[348]，主要包括关节角度、速度、角速度、质心偏移和约束等。
动作空间：一维向量[17]，关节力矩
系统奖励：reward = healthy_reward + forward_reward - ctrl_cost - contact_cost.
自定义奖励：根据人运动对应关节的正太分布取概率值+频率概率值+人关节角度与机器人对应关节角度的KL散度

ppo算法实现algorithm/ppo.py
经验回放：algorithm/replay_buffer.py

🚀 快速开始
环境要求
Python 3.9+
MuJoCo 3.3.0+
gymnasium 1.1.1
其他库版本参考requirements.txt

训练模型
mian.py

测试模型
test.py


