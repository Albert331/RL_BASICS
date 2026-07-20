import gymnasium as gym
import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense

env = gym.make('CartPole-v1')

net_input = Input(shape=(4,))
x=Dense(64,activation='relu')(net_input)
x=Dense(64,activation='relu')(x)
output = Dense(2,activation='linear')(x)    
q_net = Model(inputs=net_input,outputs=output)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
alpha=0.001
epsilon=0.1
gamma=0.99
num_episodes=500

def policy(state,explore=0.0):
    action = tf.argmax(q_net(state)[0],output_type=tf.int32)
    if tf.random.uniform(shape=(),maxval=1) <= explore:
        action = tf.random.uniform(shape=(),minval=0,maxval=2,dtype=tf.int32)
    return action    

for i in range(num_episodes):
    done = False
    state,info = env.reset()
    state = tf.convert_to_tensor([state])
    action = policy(state,epsilon)
    tr=0
    while not done:
        next_state, reward,terminated,truncated,_ = env.step(action.numpy())
        next_state = tf.convert_to_tensor([next_state])
        next_action = policy(next_state,epsilon)

        target = reward + gamma * q_net(next_state)[0][next_action]
        if done:
            target = reward

        with tf.GradientTape() as tape:
            current_q = q_net(state)[0][action]
            loss = tf.square(target - current_q)

        grads = tape.gradient(loss, q_net.trainable_weights)
        optimizer.apply_gradients(zip(grads, q_net.trainable_weights))

        state = next_state
        action = next_action
        tr+=reward
        done = terminated or truncated
    print(i+1,' ====== ',tr)    
env.close()    