import gymnasium as gym
import numpy as np

cliffenv = gym.make('CliffWalking-v1', render_mode='ansi')


q_table = np.zeros(shape=(48,4))


epsilon = 0.1
alpha=0.1
gamma=0.9
epoch  = 500

def policy(state,explore=0.0):
    action=int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(0,4))
    return action    



for i in range(epoch):
    state, info = cliffenv.reset()
    done = False
    t_reward=0
    while not done:
        action = policy(state,epsilon)
        next_state, reward, terminated, truncated, info = cliffenv.step(action)
        next_action = policy(next_state)

        q_table[state][action] += alpha * (reward + gamma * q_table[next_state][next_action] - q_table[state][action])
        state = next_state
        done = terminated or truncated
        t_reward += reward
        
    print(i+1,' ====== ',t_reward)

 
cliffenv.close()

