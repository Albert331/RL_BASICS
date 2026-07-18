import gymnasium as gym
import numpy as np

cliffenv = gym.make('CliffWalking-v1', render_mode='ansi')
state, info = cliffenv.reset()
done = False

while not done:
    print(cliffenv.render())
    action = np.random.randint(0, 4)
    print(state, '---->', action)
    state, reward, terminated, truncated, info = cliffenv.step(action)
    done = terminated or truncated

cliffenv.close()