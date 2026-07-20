import gymnasium as gym
from stable_baselines3 import A2C

env = gym.make('LunarLander-v3')
state,info= env.reset()


model = A2C('MlpPolicy',env,verbose=1)
model.learn(total_timesteps=100000)

episode = 10
eval_env = gym.make('LunarLander-v3', render_mode='human')
for ep in range(episode):
    state,info= env.reset()
    done = False

    while not done:
        
        action, _states = model.predict(state)
        state, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
env.close()