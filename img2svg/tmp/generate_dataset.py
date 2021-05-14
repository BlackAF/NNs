#%%
import numpy as np
import cairosvg
import time
import tensorflow as tf
import pathlib
import csv
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

#%%
class SVGDatasetGenerator:
    ORIGIN_CMD = -1
    LINE_CMD = -2
    CURVE_CMD = -3

    def __init__(self, log_dir=None, size=10):
        self.log_dir = log_dir
        self.size = size

        if log_dir is not None:
            pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    def generate_origin_anchor(self):
        anchor = np.random.randint(0, self.size+1, size=2)
        
        anchor_str = f'M{anchor[0]},{anchor[1]}'

        return anchor, anchor_str
    
    def generate_line_anchor(self):
        anchor = np.random.randint(0, self.size+1, size=2)
        
        anchor_str = f'L{anchor[0]},{anchor[1]}'
        
        return anchor, anchor_str
        
    def generate_curve_anchor(self):
        anchor = np.random.randint(0, self.size+1, size=6)
        
        anchor_str = f'C{anchor[0]},{anchor[1]} {anchor[2]},{anchor[3]} {anchor[4]},{anchor[5]}'
        
        return anchor, anchor_str
    
    def generate_anchor(self, command):
        if command == self.ORIGIN_CMD:
            return self.generate_origin_anchor()
        elif command == self.LINE_CMD:
            return self.generate_line_anchor()
        elif command == self.CURVE_CMD:
            return self.generate_curve_anchor()
        else:
            raise ValueError(f'Invalid command: {command}')
    
    def generate_svg(self, num_paths, num_commands, curve_only=False, line_only=False):
        anchors = []
        svg = []
        
         # Build opening element
        svg_begin = f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.size}" height="{self.size}">\n'
        svg.append(svg_begin)

        for _ in range(num_paths):
            anchors_str = []

            if curve_only:
                choices = [self.CURVE_CMD]
            elif line_only:
                choices = [self.LINE_CMD]
            else:
                choices = [self.LINE_CMD, self.CURVE_CMD]

            commands = np.random.choice(choices, size=num_commands)
            # The first command should be the origin
            commands[0] = self.ORIGIN_CMD

            # Generate anchors for each command type
            for command in commands:
                anchor, anchor_str = self.generate_anchor(command)

                anchors.append(anchor)
                anchors_str.append(anchor_str)

            # Join all the anchors together (ex M13,28 L38,2 C45,32 83,85 57,27)
            anchors_str = ' '.join(anchors_str)

            # Build the path element
            path_str = f'\t<path fill="none" stroke="#ffffff" d="{anchors_str}" />\n'
            svg.append(path_str)
            
        # Flatten the anchors into one list
        anchors = np.concatenate(anchors)
        
        # Build closing element
        svg_end = '</svg>\n'
        svg.append(svg_end)
        svg = ''.join(svg)
        
        return svg, anchors
        
                
    def save(self, batch_size, num_paths, num_commands, curve_only=False, line_only=False):
        assert batch_size > 0
        assert num_paths > 0
        assert num_commands > 1
        
        name = f'{num_paths}_paths_{num_commands}_commands_{batch_size}_samples_'
        name = name + ('line_only' if line_only else 'curve_only' if curve_only else 'mixed')

        samples = []
        labels = []

        # Generate random coordinates
        for _ in tqdm(range(batch_size)):
            svg, anchors = self.generate_svg(num_paths, num_commands, curve_only=curve_only, line_only=line_only)
            anchors = np.reshape(anchors, (-1, 2))
            # Make sure we always have 4 control points (pad with -size which will be rescaled to -1)
            anchors = np.pad(anchors, pad_width=[(0, 4-anchors.shape[0]), (0, 0)], constant_values=-self.size)

            timestamp = str(time.time()).replace('.', '')
            file_name = f'{name}-{timestamp}'

            svg_png = cairosvg.svg2png(bytestring=svg.encode('UTF-8'))
            svg_png = tf.io.decode_png(svg_png, channels=1)

            samples.append(svg_png)
            labels.append(anchors)

        # Scale between 0 and 1
        samples = np.stack(samples) / 255
        labels = np.stack(labels) / self.size

        with h5py.File(os.path.join(self.log_dir, f'{file_name}.hdf5'), 'w') as f:
            f.create_dataset('x_train', data=samples, dtype=np.float32)
            f.create_dataset('y_train', data=labels, dtype=np.float32)

#%%
start_time = time.time()

for _ in range(10):
    SVGDatasetGenerator(log_dir='datasets').save(batch_size=5000, num_paths=1, num_commands=2, curve_only=True)

end_time = time.time() - start_time
print('\nDone! - Took: ', end_time, 's')
