import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

theta = np.ones(5)

def q(s,a, theta):
    feature = np.concatenate((s,[a]))
    return np.dot(theta, feature)

alpha = 0.1
discount = 0.8
n = 30

for episodes in range(10000):
    state, _ = env.reset()
    done = False
    while not done:
        if np.random.random() < 1/n: #exploration rate
            action = env.action_space.sample()
        else:
            if q(state,1 , theta) > q(state, 0, theta):
                action = 1
            else:
                action = 0 
        
        next_state, reward, done, _, _ = env.step(action)
        best_next_action = 0 if q(next_state,0,theta) > q(next_state,1,theta) else 1
        update = alpha * (reward + discount*q(next_state, best_next_action, theta) - q(state, action, theta))
        feature = np.concatenate((state,[action]))
        theta += feature*update
        state = next_state
        if episodes % 100 == 0:
            print(f"Episode {episodes}, Theta: {theta}")
        

        
    


    

