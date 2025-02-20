import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size) 
        self.output = nn.Linear(hidden_size, output_size)  
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.activation(self.hidden(x)) 
        x = self.output(x)  
        return x

QApprox = NeuralNet(4,5,2) #we will try a hidden layer of size 4 for our function which approximates Q
criterion = nn.MSELoss()
optimizer = optim.Adam(QApprox.parameters(), lr = 0.01)

env = gym.make("CartPole-v1", render_mode="rgb_array")
discount = 0.9
n = 1

for episodes in range(1000000):

    state, _ = env.reset()
    state_tensor = torch.from_numpy(state).float() 
    done = False
    steps = 0.0

    while not done:
        steps += 1
        if np.random.random() < 1.0/n: #exploration rate
            action = env.action_space.sample()
        else:
            left_q, right_q = QApprox(state_tensor)[0], QApprox(state_tensor)[1] 
            action = 0 if left_q > right_q else 1
        
        #let us use the bellman optimality eqn to find an appropriate loss >
        next_state, reward, done, _, _ = env.step(action)
        next_state_tensor = torch.from_numpy(next_state).float()
        reward_tensor = torch.tensor([reward], dtype=torch.float32) 

        with torch.no_grad():
            next_q_values = QApprox(next_state_tensor)
            expected_reward = reward + discount * torch.max(next_q_values)  # Max Q-value for next state

        target_q = QApprox(state_tensor)[action]  # Get Q-value for the taken action
        loss = criterion(target_q, expected_reward)
        #update model parameters >
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if done and episodes % 1000 == 0:
            print(f'Trial number {episodes} failed after {steps/50} seconds')
        state_tensor = next_state_tensor
    n += 1







        
    


    

