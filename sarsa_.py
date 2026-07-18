import gymnasium as gym
import numpy as np

cliffenv = gym.make('CliffWalking-v1', render_mode='ansi')


q_table = np.zeros(shape=(48,4))

def policy(state,explore=0.0):
    action=np.argmax(q_table[state])
    if np.random.random() <= explore:
        action = int(np.random.randint(0,4))
    return action    


epsilon = 0.1
alpha=0.1
gamma=0.9
epoch  = 500


for i in range(epoch):
    state, info = cliffenv.reset()
    done = False
    t_reward=0
    action = policy(state,epsilon)
    while not done:
        next_state, reward, terminated, truncated, info = cliffenv.step(action)
        next_action = policy(next_state,epsilon)

        q_table[state][action] += alpha * (reward + gamma * q_table[next_state][next_action] - q_table[state][action])
        state = next_state
        action = next_action
        done = terminated or truncated
        t_reward += reward
        
    print(i+1,' ====== ',t_reward)

print(q_table)
policy_actions = np.argmax(q_table, axis=1)
print(policy_actions.reshape(4, 12))    
cliffenv.close()

