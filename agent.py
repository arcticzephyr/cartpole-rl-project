import torch
import torch.nn as nn
import torch.optim as optim
import random

#First we simply define a simple neural net

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size) 
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)  
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.output(x)  
        return x

class DQNAgent():
    def __init__(self, state_dim, action_dim, batch_size, lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.model = NeuralNet(state_dim, 128, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)
        self.criterion = nn.MSELoss()
    def action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return torch.argmax(self.model(state)).item()
    def train(self, replay_buffer):
        if replay_buffer.len() < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_q_values = torch.max(self.model(next_states), dim=1)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)#bellman equation <--

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        self.optimizer.zero_grad()
        loss = self.criterion(current_q, target_q)
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        
        

