#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cairosvg
import pathlib
import time

from tensorflow.keras.layers import GRU
from scipy.special import comb

tf.config.list_physical_devices('GPU')

#%%
control_points = np.random.randint(0, 101, size=(4,2))
print('control points:', control_points)

#%%
svg_begin = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">\n'
rnd = lambda: np.random.randint(0, 101)
# path_str = f'\t<path fill="none" stroke="#ffffff" d="M {rnd()},{rnd()} Q {rnd()},{rnd()},{rnd()},{rnd()}" />\n'
cp = control_points.flatten()
path_str = f'<path fill="none" stroke="#ffffff" d="M {cp[0]},{cp[1]} C {cp[2]},{cp[3]} {cp[4]},{cp[5]} {cp[6]},{cp[7]}" />'
svg_end = '</svg>\n'

svg = ''.join([svg_begin, path_str, svg_end])
print('svg', svg)

img = cairosvg.svg2png(svg.encode('utf-8'))
img = tf.io.decode_jpeg(img, channels=1)

plt.imshow(img, origin='lower')
plt.show()

#%%
def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(control_points, smoothness):
    x = control_points[:, 0]
    y = control_points[:, 1]

    num_control_points = len(control_points)

    t = np.linspace(0.0, 1.0, smoothness)

    polynomial_array = np.array([bernstein_poly(i, num_control_points-1, t) for i in range(num_control_points)])
    
    x_bezier = np.array([int(round(i)) for i in np.dot(x, polynomial_array)])
    y_bezier = np.array([int(round(i)) for i in np.dot(y, polynomial_array)])

    return np.stack((x_bezier, y_bezier), axis=1)

curve = bezier_curve(control_points, 100)
print('length', len(curve))
curve = set(tuple(point) for point in curve)
curve = np.array(list(curve))
# curve = np.unique(curve, axis=0)
print('unique length', len(curve))

bezier_img = np.zeros_like(np.squeeze(img))
# print(curve.shape)
bezier_img[curve[:, 1], curve[:, 0]] = 255

print('points', np.transpose(bezier_img.nonzero()).shape)

plt.imshow(bezier_img, origin='lower')
plt.show()

#%%

b = np.array([(1,2), (3,4), (1,2), (1,2), (3,5), (3,4)])
# b = [(1,2), (3,4)]

print(type(b))

a = set(tuple(c) for c in b)

print(a)

#%%
class DataGenerator:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir

        if log_dir is not None:
            pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    def generate_curves(self, batch_size):
        assert batch_size > 0

        train_x = []
        train_y = []

        for _ in range(batch_size):
            control_points = np.random.randint(0, 101, size=(4,2))
            smoothness = np.random.randint(10, 150)

            points = self.bezier_curve(control_points, smoothness)

            train_x.append(points)
            train_y.append(control_points)

        # Combine samples
        train_x = np.array(train_x, dtype=object)
        train_y = np.array(train_y, dtype=float)

        # Normalize to 0.0 - 1.0 range
        train_x /= 100
        train_y /= 100

        np.savez(str(pathlib.Path(self.log_dir, 'dataset.npz')), train_x=train_x, train_y=train_y)

    def bernstein_poly(self, i, n, t):
        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bezier_curve(self, control_points, smoothness):
        x = control_points[:, 0]
        y = control_points[:, 1]

        num_control_points = len(control_points)

        t = np.linspace(0.0, 1.0, smoothness)

        polynomial_array = np.array([self.bernstein_poly(i, num_control_points-1, t) for i in range(num_control_points)])
        
        x_bezier = np.array([int(round(i)) for i in np.dot(x, polynomial_array)])
        y_bezier = np.array([int(round(i)) for i in np.dot(y, polynomial_array)])

        points = np.stack((x_bezier, y_bezier), axis=1)

        # Remove duplicates
        points = set(tuple(point) for point in points)
        points = np.array(list(points))

        return points

#%%
print('Starting!')
start_time = time.time()

DataGenerator('.').generate_curves(batch_size=100)

end_time = time.time() - start_time
print('Done! - Took: ', end_time, 's')



