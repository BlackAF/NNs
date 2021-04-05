#%%
import numpy as np
import cairosvg
import time
import tensorflow as tf
import pathlib
import csv
import matplotlib.pyplot as plt

#%%
class SVGDatasetGenerator:
    ORIGIN_CMD = -1
    LINE_CMD = -2
    CURVE_CMD = -3

    def __init__(self, log_dir=None):
        self.log_dir = log_dir

        if log_dir is not None:
            pathlib.Path(log_dir, 'svgs').mkdir(parents=True, exist_ok=True)
            pathlib.Path(log_dir, 'imgs').mkdir(parents=True, exist_ok=True)
    
    def generate_origin_anchor(self):
        anchor = np.random.randint(0, 101, size=3)
        # Set the first value to represent the command type
        anchor[0] = self.ORIGIN_CMD
        
        anchor_str = f'M{anchor[1]},{anchor[2]}'

        return anchor, anchor_str
    
    def generate_line_anchor(self):
        anchor = np.random.randint(0, 101, size=3)
        # Set the first value to represent the command type
        anchor[0] = self.LINE_CMD    
        
        anchor_str = f'L{anchor[1]},{anchor[2]}'
        
        return anchor, anchor_str
        
    def generate_curve_anchor(self):
        anchor = np.random.randint(0, 101, size=7)
        # Set the first value to represent the command type
        anchor[0] = self.CURVE_CMD
        
        anchor_str = f'C{anchor[1]},{anchor[2]} {anchor[3]},{anchor[4]} {anchor[5]},{anchor[6]}'
        
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
    
    def generate_svg(self, num_paths, num_anchors, curve_only=False, line_only=False):
        anchors = []
        svg = []
        
         # Build opening element
        svg_begin = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">\n'
        svg.append(svg_begin)

        for _ in range(num_paths):
            anchors_str = []

            if curve_only:
                choices = [self.CURVE_CMD]
            elif line_only:
                choices = [self.LINE_CMD]
            else:
                choices = [self.LINE_CMD, self.CURVE_CMD]

            commands = np.random.choice(choices, size=num_anchors)
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
            path_str = f'\t<path fill="none" stroke="#00ff00" d="{anchors_str}" />\n'
            svg.append(path_str)
            
        # Flatten the anchors into one list
        anchors = np.concatenate(anchors)
        
        # Build closing element
        svg_end = '</svg>\n'
        svg.append(svg_end)
        svg = ''.join(svg)
        
        return svg, anchors
        
                
    def save(self, batch_size, num_paths, num_anchors, curve_only=False, line_only=False):
        assert batch_size > 0
        assert num_paths > 0
        assert num_anchors > 1
        
        name = f'{num_paths}_paths_{num_anchors}_anchors_'
        name = name + ('line_only' if line_only else 'curve_only' if curve_only else 'mixed')

        rows = []

        # Generate random coordinates
        for _ in range(batch_size):
            paths = []
            svg = []
            
            svg, paths = self.generate_svg(num_paths, num_anchors, curve_only=curve_only, line_only=line_only)
            
            timestamp = str(time.time()).replace('.', '')
            file_name = f'{name}-{timestamp}'

            # Write the image to disk
            self.save_img(svg, file_name)

            anchors_scaled = list(paths.flatten() / 100)
            rows.append([file_name, ','.join(map(str, anchors_scaled))])
        
        with open(pathlib.Path(self.log_dir, 'dataset.csv'), 'w') as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerows(rows)

    def save_svg(self, svg, file_name):
        svg_path = pathlib.Path(self.log_dir, 'svgs', f'{file_name}.svg')
        with open(svg_path, 'w') as f:
            f.write(svg)

    def save_img(self, svg, file_name):
        image_path = str(pathlib.Path(self.log_dir, 'imgs', f'{file_name}.png'))
        cairosvg.svg2png(bytestring=svg.encode('UTF-8'), write_to=image_path)

#%%
print('Starting!')
start_time = time.time()

SVGDatasetGenerator(log_dir='datasets/mixed/train').save(batch_size=5, num_paths=1000, num_anchors=1000)

end_time = time.time() - start_time
print('Done! - Took: ', end_time, 's')