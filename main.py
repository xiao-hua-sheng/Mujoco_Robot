import os.path

import gymnasium as gym
import torch
import time

from typing import List, Tuple
from agent.agent import PPOAgent
from algorithm.ppo import PPO
from reward.reward import HumanoidReward
from algorithm.replay_buffer import ExperienceReplayBuffer
from torch.utils.tensorboard import SummaryWriter


class PPOTrainer:
    def __init__(self, env_name: str = "Humanoid-v5", config: dict = None, load_path: str = ""):
        # 初始化环境
        self.render = False
        self.reward_source = True
        self.env_name = env_name
        self.save_path = "model"
        if self.render:
            self.env = gym.make(env_name, render_mode='human')
        else:
            self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # 初始化组件
        self.agent = PPOAgent(self.state_dim, self.action_dim)
        if load_path:
            checkpoint = torch.load(load_path)
            self.agent.load_state_dict(checkpoint)
        self.ppo = PPO(
            agent=self.agent,
            gamma=config.get('gamma', 0.99),
            clip_epsilon=config.get('clip_epsilon', 0.2),
            entropy_coef=config.get('entropy_coef', 0.01)
        )
        self.reward_calculator = HumanoidReward(**config.get('reward_params', {}))

        # 训练参数
        self.max_episodes = config.get('max_episodes', 1000)
        self.max_steps = config.get('max_steps', 2000)
        self.update_interval = config.get('update_interval', 4800)
        self.batch_size = 64

        self.replay_buff = ExperienceReplayBuffer(max_size=2000, batch_size=self.batch_size)
        self.writer = SummaryWriter(f'runs/{self.env_name}_reward_{int(time.time())}')


    def collect_trajectory(self) -> Tuple[List, List, List, List, List]:
        """收集单条轨迹数据"""
        states, actions, rewards, dones, old_log_probs = [], [], [], [], []
        state, _ = self.env.reset()
        self.reward_calculator.reset()

        for _ in range(self.max_steps):
            with torch.no_grad():
                action, old_log_prob = self.agent.get_action(state)

            next_state, raw_reward, done, truncated, _ = self.env.step(action.numpy())
            if self.reward_source:
                processed_reward = float(raw_reward)
            else:
                processed_reward = self.reward_calculator.compute(
                    state, action.numpy(), next_state, raw_reward, done or truncated
                )

            # 存储转换数据
            states.append(state)
            actions.append(action.numpy())
            rewards.append(processed_reward)
            dones.append(done or truncated)
            old_log_probs.append(old_log_prob.numpy().flatten())

            state = next_state
            # if done or truncated:
            #     break

            if self.render:
                self.env.render()
        return states, actions, rewards, dones, old_log_probs

    def train(self):
        """主训练循环"""
        for episode in range(self.max_episodes):
            # 数据收集
            states, actions, rewards, dones, old_log_probs = self.collect_trajectory()
            # 日志记录
            episode_reward = sum(rewards) / len(rewards)
            print(f"Episode {episode + 1}/{self.max_episodes}, Total Reward: {episode_reward:.1f}")
            self.writer.add_scalar('Episode/Reward', episode_reward, episode)
            # 保存轨迹并从经验池中采样
            self.replay_buff.add(states, actions, rewards, dones, old_log_probs)

            states, actions, rewards, dones, old_log_probs = self.replay_buff.sample(self.batch_size)
            # 策略更新
            self.ppo.update(
                states=states,
                actions=actions,
                rewards=rewards,
                dones=dones,
                old_log_probs=old_log_probs,
                episode=episode)

            if (episode+1) % 10000 == 0:
                # 保存模型
                model_name = os.path.join(self.save_path, "humanoid_ppo_{}w.pth".format((episode+1) // 10000))
                torch.save(self.agent.state_dict(), model_name)
        self.env.close()
        self.writer.close()
        self.ppo.writer.close()



if __name__ == "__main__":
    config = {
        'gamma': 0.99,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.001,
        'original_coef': 0.1,
        'max_episodes': 100000,
        'max_steps': 200,
        'reward_params': {
            'height_coef': 0.15,
            'energy_coef': 0.01,
            'survive_bonus': 0.3,
            'target_height': 1.4
        }
    }

    trainer = PPOTrainer(env_name="Humanoid-v5", config=config, load_path="model/humanoid_ppo_4w_old.pth")
    trainer.train()