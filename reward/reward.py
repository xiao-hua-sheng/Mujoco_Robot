import numpy as np
from typing import Optional


class HumanoidReward:
    """可定制的奖励计算器"""

    def __init__(self,
                 height_coef: float = 0.1,
                 velocity_coef: float = 0.05,
                 energy_coef: float = 0.001,
                 original_coef: float = 0.1,
                 survive_bonus: float = 0.2,
                 target_height: float = 1.4,
                 max_episode_steps: int = 1000):

        # 奖励系数
        self.height_coef = height_coef
        self.velocity_coef = velocity_coef
        self.energy_coef = energy_coef
        self.survive_bonus = survive_bonus
        self.original_coef = original_coef

        # 目标参数
        self.target_height = target_height
        self.max_episode_steps = max_episode_steps

        # 状态跟踪
        self.prev_action: Optional[np.ndarray] = None
        self.prev_position: Optional[np.ndarray] = None
        self.step_count: int = 0

    def reset(self):
        """重置episode相关的状态"""
        self.prev_action = None
        self.prev_position = None
        self.step_count = 0

    def compute(self,
                state: np.ndarray,
                action: np.ndarray,
                next_state: np.ndarray,
                original_reward: float,
                done: bool) -> float:
        """
        计算综合奖励值
        参数：
            state: 当前状态 (s_t)
            action: 执行的动作 (a_t)
            next_state: 下一状态 (s_{t+1})
            original_reward: 环境原始奖励
            done: 是否终止
        返回：
            处理后的奖励值
        """
        reward = original_reward * self.original_coef

        # 1. 高度奖励
        current_height = next_state[0]
        height_reward = self._calculate_height_reward(current_height)

        # 2. 速度奖励
        velocity_reward = self._calculate_velocity_reward(state, next_state)

        # 3. 能量效率惩罚
        energy_penalty = self._calculate_energy_penalty(action)

        # 4. 存活奖励
        survival_reward = self.survive_bonus if not done else 0

        # 5. 进度奖励（可选）
        progress_reward = self._calculate_progress_reward()

        # 组合奖励
        total_reward = (
                reward
                + height_reward * self.height_coef
                + velocity_reward * self.velocity_coef
                - energy_penalty * self.energy_coef
                + survival_reward
        )

        # 更新状态
        self.prev_action = action.copy()
        self.prev_position = state[:3].copy()
        self.step_count += 1

        return float(total_reward)

    def _calculate_height_reward(self, height: float) -> float:
        """计算基于高度的奖励"""
        return max(0.0, height - self.target_height)

    def _calculate_velocity_reward(self,
                                   current_state: np.ndarray,
                                   next_state: np.ndarray) -> float:
        """计算基于水平速度的奖励"""
        if self.prev_position is None:
            return 0.0
        dx = next_state[0] - current_state[0]  # x轴方向位移
        dy = next_state[1] - current_state[1]  # y轴方向位移
        horizontal_speed = np.sqrt(dx ** 2 + dy ** 2) / 0.05  # 假设时间步0.05秒
        return horizontal_speed

    def _calculate_energy_penalty(self, action: np.ndarray) -> float:
        """计算能量消耗惩罚"""
        if self.prev_action is None:
            return 0.0
        action_diff = np.mean(np.square(action - self.prev_action))
        return action_diff

    def _calculate_progress_reward(self) -> float:
        """计算训练进度奖励"""
        progress = self.step_count / self.max_episode_steps
        return 0.5 * (1 + np.cos(np.pi * progress))  # 余弦衰减奖励