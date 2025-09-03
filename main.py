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
from tool.config import ConfigLoader


class PPOTrainer:
    def __init__(self, configs: dict):
        # 初始化环境
        self.render = False
        self.reward_source = True
        self.env_name = configs["environment"]["name"]
        self.pre_model_path = configs["training"]["pre_model_path"]
        self.save_path = configs["training"]["output_dir"]
        if self.render:
            self.env = gym.make(self.env_name, render_mode='human')
        else:
            self.env = gym.make(self.env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.agent_network = configs["environment"]["agent_network"]

        # 初始化组件
        self.agent = PPOAgent(self.state_dim, self.action_dim, hidden_dim=self.agent_network)
        if self.pre_model_path:
            checkpoint = torch.load(self.pre_model_path)
            self.agent.load_state_dict(checkpoint)
        self.ppo = PPO(
            agent=self.agent,
            gamma=configs["algorithm"].get('gamma', 0.99),
            clip_epsilon=configs["algorithm"].get('clip_epsilon', 0.2),
            entropy_coef=configs["algorithm"].get('entropy_coef', 0.01),
            lr=configs["algorithm"].get('learning_rate', 1e-4)
        )
        self.reward_calculator = HumanoidReward(**configs['algorithm'].get('reward_params', {}))

        # 训练参数
        self.max_episodes = configs["training"].get('max_episodes', 1000)
        self.max_steps = configs["algorithm"].get('max_steps', 2000)
        self.update_interval = configs["algorithm"].get('update_interval', 4800)
        self.batch_size = configs['training']['batch_size']

        self.buffer_size = configs['replay_buffer']['buffer_size']

        self.replay_buff = ExperienceReplayBuffer(max_size=self.buffer_size, batch_size=self.batch_size)
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
                model_name = os.path.join(self.save_path, "{}_ppo_{}w.pth".format(self.env_name[:-3], (episode+1) // 10000))
                torch.save(self.agent.state_dict(), model_name)
        self.env.close()
        self.writer.close()
        self.ppo.writer.close()



if __name__ == "__main__":
    config_loader = ConfigLoader("tool/config.yaml")
    config = config_loader.load_config()

    trainer = PPOTrainer(configs=config)
    trainer.train()