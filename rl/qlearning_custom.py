#%%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import cairosvg
import io

from PIL import Image
from enum import Enum

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

svg = SVG(width=10, height=10)

svg.add_line((.1, .5), (.9, .5), color=np.random.randint(0, 256, size=3).tolist())
# svg.add_line((.10, .40), (.58, .37), color=np.random.randint(0, 256, size=3).tolist())
# svg.add_quadratic_bezier((.93, .16), (.75, .64), (.23, .17), color=np.random.randint(0, 256, size=3).tolist())
# svg.add_cubic_bezier((.23, .04), (.18, .64), (.85, .08), (.32, .57), color=np.random.randint(0, 256, size=3).tolist())

img = svg.to_image()

print('svg', svg)
print('image', img)

plt.rcParams['axes.facecolor'] = 'black'
plt.imshow(img, origin='lower')


# %%
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
            return self.state, reward, done

        # If the agent landed on a target remove it from the state
        new_targets = tuple(t for t in targets if t != new_agent_pos)

        # If the number of targets changed, the agent found a target so reward is 0 otherwise it is -1
        reward = self.TARGET_REWARD if len(new_targets) != len(targets) else self.MOVE_PENALTY

        # There are no more targets so the agent is done
        if len(new_targets) == 0:
            done = True

        # Update the new state
        self.state = (new_agent_pos, new_targets)

        return self.state, reward, done

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
        img = np.zeros((self.width, self.height))

        agent_pos, targets = self.state

        # Draw the targets if there are any left
        if len(targets):
            target_rows, target_cols = np.transpose(targets)
            img[target_rows, target_cols] = 150

        # Draw the agent
        img[agent_pos] = 255

        plt.rcParams['axes.facecolor'] = 'black'
        plt.imshow(img, cmap='gray', origin='lower')
        plt.show()

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

        return self.state

#%%
class TrainManager:
    def __init__(self):
        self.env = Environment()
        self.default_q_values = lambda: np.random.uniform(-2, 0, size=self.env.NUM_ACTIONS)
        self.q_table = dict()
        
    def train(self, episodes, lr, discount, epsilon, render=False):
        start_epsilon_decay = 1
        end_epsilon_decay = episodes // 2
        epsilon_decay = epsilon / (end_epsilon_decay - start_epsilon_decay)
        
        for episode in range(episodes):
            current_state = self.env.reset()
            done = False

            if episode % 500 == 0:
                print('Episode: ', episode)

            while not done:
                # Get the q values for the current state
                current_q_values = self.q_table.get(current_state, self.default_q_values())
                
                # Choose action either randomly or bassed on q values
                if np.random.uniform() < epsilon:
                    action = self.env.sample_action()
                else:
                    action = np.argmax(current_q_values)

                # Perform the action
                new_state, reward, done = self.env.step(action)

                if done:
                    new_q = 0
                else:
                    # Get current and next q values
                    current_q = current_q_values[action]
                    max_future_q = np.max(self.q_table.get(new_state, self.default_q_values()))

                    # Calculate new q value
                    new_q = (1 - lr) * current_q + lr * (reward + discount * max_future_q)

                # Update q value
                current_q_values[action] = new_q

                self.q_table[current_state] = current_q_values

                current_state = new_state

                if start_epsilon_decay <= episode <= end_epsilon_decay:
                    epsilon -= epsilon_decay

                if render:
                    self.env.render()

#%%
tm = TrainManager()
tm.train(episodes=5000, lr=0.1, discount=0.98, epsilon=0.5)
print('--------DONE--------')

# %%
tm.train(episodes=1, lr=0.1, discount=0.95, epsilon=0.5, render=True)

# print(list(tm.q_table.values())[:500])
