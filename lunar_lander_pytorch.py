import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Qnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,4)
        

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))   
        x = self.fc3(x)
        return x
    
   
q_net = Qnet()
target_net = Qnet()
target_net.load_state_dict(q_net.state_dict())   
optimizer = optim.Adam(q_net.parameters(), lr=0.001)


env = gym.make('LunarLander-v3',render_mode = 'human')
state,info= env.reset()
done = False


def policy(state, explore=0.0):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = torch.argmax(q_net(state_tensor)).item()
    if np.random.random() <= explore:
        action = np.random.randint(0, 4)
    return action


optimizer = optim.Adam(q_net.parameters(), lr=0.001)
epsilon = 0.1
gamma = 0.9
epoch = 500

for i in range(epoch):
    state,info= env.reset()
    action = policy(state,epsilon)
    done = False
    tr=0
    while not done:
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_action = policy(next_state,epsilon)

        with torch.no_grad():
            if done:
                target = torch.tensor(reward,dtype = torch.float32)
            else:
                next_state_tensor =  torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                target = reward + gamma * q_net(next_state_tensor)[0][next_action]        

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        current_q = q_net(state_tensor)[0][action]

        loss = (target - current_q) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        action = next_action
        tr += reward

    print(i+1, ' ====== ', tr)
        