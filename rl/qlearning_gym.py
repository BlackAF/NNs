#%%
import gym
import numpy as np
import matplotlib.pyplot as plt

num_episodes = 20000
num_timesteps = 1
discount = 0.95
lr = 0.1
show_every = 500
observation_size = [20, 20]
epsilon = 0.5
start_epsilon_decay = 1
end_epsilon_decay = num_episodes // 2
epsilon_decay = epsilon / (end_epsilon_decay - start_epsilon_decay)

env = gym.make('MountainCar-v0')

discrete_observation_size = (env.observation_space.high - env.observation_space.low) / observation_size

q_table = np.random.uniform(-2, 0, size=(observation_size + [env.action_space.n]))

# print('discrete observation size', discrete_observation_size)

rewards = []
logs = {
    'episodes': [],
    'avg': [],
    'min': [],
    'max': []
}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_observation_size
    return tuple(discrete_state.astype(np.int))

for episode in range(num_episodes):
    done = False
    discrete_state = get_discrete_state(env.reset())
    total_rewards = 0

    render = episode % show_every == 0

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        
        total_rewards += reward

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[(*discrete_state, action)]
            new_q = (1 - lr) * current_q + lr * (reward + discount * max_future_q)
            q_table[(*discrete_state, action)] = new_q
        elif new_state[0] >= env.goal_position:
            # print('Reached goal at episode: ', episode)
            q_table[(*discrete_state, action)] = 0

        discrete_state = new_discrete_state

    rewards.append(total_rewards)

    if start_epsilon_decay <= episode <= end_epsilon_decay:
        epsilon -= epsilon_decay

    if render:
        # Get the last X episodes
        episodes_batch = rewards[-show_every:]
        logs['episodes'].append(episode)
        logs['avg'].append(np.mean(episodes_batch))
        logs['min'].append(min(episodes_batch))
        logs['max'].append(max(episodes_batch))
        print(f'Episode {episode} - Min: {logs["min"][-1]} - Avg: {logs["avg"][-1]} - Max: {logs["max"][-1]}')
        # env.render()

plt.plot(logs['episodes'], logs['avg'], label='avg')
plt.plot(logs['episodes'], logs['min'], label='min')
plt.plot(logs['episodes'], logs['max'], label='max')
plt.legend()
plt.show()

env.close()

print('observation size', env.observation_space.low, env.observation_space.high)
