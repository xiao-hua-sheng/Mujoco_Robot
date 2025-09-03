import gymnasium as gym
import torch

from agent.agent import PPOAgent
from tool.config import ConfigLoader


if __name__ == "__main__":
    env_name = "Pusher-v5"
    load_path = "models/Pusher_ppo_9w.pth"
    config_loader = ConfigLoader("tool/config.yaml")
    configs = config_loader.load_config()

    env = gym.make(env_name, render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim, configs["environment"]["agent_network"])
    checkpoint = torch.load(load_path)
    agent.load_state_dict(checkpoint)

    state, _ = env.reset()
    while True:
        with torch.no_grad():
            action, old_log_prob = agent.get_action(state)

        next_state, raw_reward, done, truncated, _ = env.step(action.numpy())
        state = next_state
        env.render()
