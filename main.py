from agent import DQNAgent
from replay_buffer import ReplayBuffer
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
from collections import deque

env = gym.make("CartPole-v1", render_mode='rgb_array')
video_env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda ep: ep % 500 == 0)
agent = DQNAgent(4,2,128)
replay_buffer = ReplayBuffer()
rew_buffer = deque(maxlen = 1000)

for episodes in range(100000):

    state, _ = video_env.reset()
    done = False
    episode_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = agent.action(state_tensor)
        next_state, reward, done, _, _ = video_env.step(action)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        replay_buffer.push(state_tensor, action, reward, next_state_tensor, done)
        agent.train(replay_buffer)
        state = next_state
        episode_reward += reward
    
    rew_buffer.append(episode_reward)
    episode_reward = 0

    if episodes % 1000 == 0: 
        average = np.mean(rew_buffer)
        print(average)
        


