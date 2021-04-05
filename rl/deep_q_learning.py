#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import io
import cairosvg
import random

from PIL import Image
from enum import Enum
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar

tf.get_logger().setLevel('ERROR')
tf.config.list_physical_devices('GPU')
#%%
class SVG:
    class Mode(Enum):
        RGB = 'RGB'
        RGBA = 'RGBA'
        GRAY = 'L'

    def __init__(self, width=100, height=100):
        assert width >= 0 and height >= 0

        self.svg_buffer = io.BytesIO()
        self.width = width
        self.height = height

    def add_line(self, start, end, color=(255, 255, 255)):
        assert len(color) == 3
        start = self.get_scaled_point(start)
        end = self.get_scaled_point(end)

        color = ','.join(map(str, color))
        path = f'\t<path fill="none" stroke="rgb({color})" d="M{start[0]},{start[1]} L{end[0]},{end[1]}" />\n'
        self.svg_buffer.write(path.encode('UTF-8'))

    def add_quadratic_bezier(self, start, end, control_point, color=(255, 255, 255)):
        assert len(color) == 3
        start = self.get_scaled_point(start)
        end = self.get_scaled_point(end)
        control_point = self.get_scaled_point(control_point)

        color = ','.join(map(str, color))
        path = f'\t<path fill="none" stroke="rgb({color})" d="M{start[0]},{start[1]} Q{end[0]},{end[1]} {control_point[0]},{control_point[1]}" />\n'
        self.svg_buffer.write(path.encode('UTF-8'))

    def add_cubic_bezier(self, start, end, control_point, control_point2, color=(255, 255, 255)):
        assert len(color) == 3
        start = self.get_scaled_point(start)
        end = self.get_scaled_point(end)
        control_point = self.get_scaled_point(control_point)
        control_point2 = self.get_scaled_point(control_point2)

        color = ','.join(map(str, color))
        path = f'\t<path fill="none" stroke="rgb({color})" d="M{start[0]},{start[1]} C{end[0]},{end[1]} {control_point[0]},{control_point[1]} {control_point2[0]},{control_point2[1]}" />\n'
        self.svg_buffer.write(path.encode('UTF-8'))

    def to_image(self, mode=None):
        mode = mode or SVG.Mode.RGB
        assert isinstance(mode, SVG.Mode)

        img = cairosvg.svg2png(bytestring=str(self).encode('UTF-8'))
        img = io.BytesIO(img)
        img = Image.open(img)        
        img = img.convert(mode.value)
        img = np.asarray(img)

        return img

    def get_scaled_point(self, point):
        assert len(point) == 2
        assert 0 <= point[0] <= 1 and 0 <= point[1] <= 1

        x = int(round(point[0] * self.width))
        y = int(round(point[1] * self.height))

        return (x, y)


    def plot_img(self):
        plt.rcParams['axes.facecolor'] = 'black'
        plt.imshow(self.to_image(), origin='lower')
        plt.show()

    def __str__(self):
        svg_begin = f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}">\n'
        svg_body = self.svg_buffer.getvalue().decode('UTF-8')
        svg_end = '</svg>\n'

        return ''.join([svg_begin, svg_body, svg_end])

# svg = SVG(width=10, height=10)

# svg.add_line((.1, .5), (.9, .5), color=np.random.randint(0, 256, size=3).tolist())
# svg.add_line((.10, .40), (.58, .37), color=np.random.randint(0, 256, size=3).tolist())
# svg.add_quadratic_bezier((.93, .16), (.75, .64), (.23, .17), color=np.random.randint(0, 256, size=3).tolist())
# svg.add_cubic_bezier((.23, .04), (.18, .64), (.85, .08), (.32, .57), color=np.random.randint(0, 256, size=3).tolist())

# img = svg.to_image()

# print('svg', svg)
# print('image', img)

# plt.rcParams['axes.facecolor'] = 'black'
# plt.imshow(img, origin='lower')

#%%
class Environment:
    MOVE_PENALTY = -1
    TARGET_REWARD = 10
    NUM_ACTIONS = 4

    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height

    def move_agent(self, move_by):
        assert len(move_by) == 2

        done = False
        agent_pos, targets = self.state

        # Update the new agent's position
        new_agent_pos = (agent_pos[0] + move_by[0], agent_pos[1] + move_by[1])

        # Don't update if the agent's new position is out of range
        if (not 0 <= new_agent_pos[0] < self.height) or (not 0 <= new_agent_pos[1] < self.width):
            reward = self.MOVE_PENALTY
            return self.to_image(), reward, done

        # If the agent landed on a target remove it from the state
        new_targets = tuple(t for t in targets if t != new_agent_pos)

        # If the number of targets changed, the agent found a target so reward is 0 otherwise it is -1
        reward = self.TARGET_REWARD if len(new_targets) != len(targets) else self.MOVE_PENALTY

        # There are no more targets so the agent is done
        if len(new_targets) == 0:
            done = True

        # Update the new state
        self.state = (new_agent_pos, new_targets)

        return self.to_image(), reward, done

    def step(self, action):
        if not hasattr(self, 'state'):
            raise ValueError('State must be initialized before making a step.')

        # UP
        if action == 0:
            return self.move_agent(move_by=(0, 1))
        # RIGHT
        elif action == 1:
            return self.move_agent(move_by=(1, 0))
        # DOWN
        elif action == 2:
            return self.move_agent(move_by=(0, -1))
        # LEFT
        elif action == 3:
            return self.move_agent(move_by=(-1, 0))

        raise ValueError(f'"{action}" is not a valid action.')

    def sample_action(self):
        return np.random.randint(0, 4)

    def render(self):
        img = self.to_image()

        plt.rcParams['axes.facecolor'] = 'black'
        plt.imshow(img, cmap='gray', origin='lower')
        plt.show()

    def to_image(self):
        img = np.zeros((self.width, self.height), dtype=np.float32)

        agent_pos, targets = self.state

        # Draw the targets if there are any left
        if len(targets):
            target_rows, target_cols = np.transpose(targets)
            img[target_rows, target_cols] = 0.5

        # Draw the agent
        img[agent_pos] = 1.0
        img = np.expand_dims(img, axis=-1)

        return img

    def reset(self):
        # Build the svg
        self.svg = SVG(width=self.width, height=self.height)
        start = np.random.randint(0, 101, size=2) / 100
        end = np.random.randint(0, 101, size=2) / 100
        self.svg.add_line(start, end)

        # Get the coordinates of each pixel
        img = self.svg.to_image(SVG.Mode.GRAY)
        targets = np.transpose(np.nonzero(img))
        targets = tuple(map(tuple, targets))

        # Get the position of the agent
        x = np.random.randint(0, self.height)
        y = np.random.randint(0, self.width)
        agent_pos = (x, y)

        self.state = (agent_pos, targets)

        return self.to_image()

#%%
class TrainManager:
    def __init__(self):
        self.env = Environment()
        self.replay_memory = deque(maxlen=C.REPLAY_MEMORY_LEN)

        self.epsilon = C.START_EPSILON

        self.model = self.load_model()
        self.target_model = self.load_model()
        self.update_target_model()

        self.progbar = Progbar(C.EPISODES, unit_name='Episode')

    def load_model(self):
        model = Sequential()

        model.add(Input((10, 10, 1)))

        model.add(Conv2D(128, 3, activation='relu', padding='same'))
        model.add(Conv2D(128, 3, activation='relu', padding='same'))
        model.add(MaxPooling2D(strides=2, padding='same'))

        model.add(Conv2D(64, 3, activation='relu', padding='same'))
        model.add(Conv2D(64, 3, activation='relu', padding='same'))
        model.add(MaxPooling2D(strides=2, padding='same'))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.env.NUM_ACTIONS))

        model.compile(Adam(lr=C.INIT_LR), loss='mse')

        return model

    def get_qs(self, state):
        state = np.expand_dims(state, axis=0)
        qs = self.model.predict(state)[0]

        return qs

    def get_action(self, state):
        # Choose action either randomly or bassed on q values
        if np.random.uniform() < self.epsilon:
            action = self.env.sample_action()
        else:
            action = np.argmax(self.get_qs(state))

        return action

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self):
        for episode in range(C.EPISODES):
            # Navigate through the environment until we finish the episode
            self.train_episode()

            # Update the target's model every x episode
            if episode % C.UPDATE_TARGET_EVERY == 0:
                self.update_target_model()

            if self.epsilon > C.END_EPSILON:
                self.epsilon *= C.EPSILON_DECAY
                self.epsilon = max(self.epsilon, C.END_EPSILON)

            self.progbar.update(current=episode+1)

    def train_episode(self):
        done = False
        episode_rewards = 0
        current_state = self.env.reset()

        while not done:
            # Choose an action
            action = self.get_action(current_state)                

            # Perform the action
            new_state, reward, done = self.env.step(action)

            # Update the replay memory
            self.replay_memory.append((current_state, action, reward, new_state, done))

            # Update the model that produces the q values with back propagation
            self.update_q_values()

            if C.RENDER:
                self.env.render()

            episode_rewards += reward
            current_state = new_state

    def update_q_values(self):
        if len(self.replay_memory) < C.MIN_REPLAY_MEMORY:
            return

        # Choose a random batch from the past replays
        random_replays = random.sample(self.replay_memory, C.BATCH_SIZE)
        
        # random_replays is of shape [batch_size, tuple of replay values]
        # so by transposing we can get an array of each column
        current_states, actions, rewards, new_states, dones = np.transpose(random_replays)

        # Adjust types
        current_states = np.stack(current_states)
        new_states = np.stack(new_states)
        actions = actions.astype(np.int32)
        dones = dones.astype(np.bool)

        # Get the current q values as well as the future q values
        current_qs = self.model.predict(current_states)
        future_qs = self.target_model.predict(new_states)

        # Get the max q value for each sample in the batch
        max_future_qs = np.max(future_qs, axis=1)

        # Calculate the new q values
        new_qs = rewards + C.DISCOUNT * max_future_qs
        # Samples that were marked as done should set their new q value to the reward
        new_qs[dones] = rewards[dones]

        # For each sample in the batch update the q value corresponding to the action that was taken
        current_qs[range(C.BATCH_SIZE), actions] = new_qs

        self.model.fit(x=current_states, y=current_qs, batch_size=C.BATCH_SIZE, verbose=0, shuffle=False)

tm = TrainManager()
tm.train()

#%%
class C:
    REPLAY_MEMORY_LEN = 50000
    MIN_REPLAY_MEMORY = 20000
    BATCH_SIZE = 32
    DISCOUNT = 0.99
    START_EPSILON = 1.0
    END_EPSILON = 0.1
    EPSILON_DECAY = 0.995
    UPDATE_TARGET_EVERY = 5
    EPISODES = 5000
    RENDER = False
    INIT_LR = 1e-03
