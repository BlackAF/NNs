#%%
import matplotlib.pyplot as plt
import cairosvg
import tensorflow as tf

#%%
def parse_control_points(*control_points):
    def parse(control_point):
        assert len(control_point) == 2
        control_point = tf.expand_dims(control_point, axis=-1)

        return control_point

    return [parse(cp) for cp in control_points]

def line(control_points, smoothness):
    p1, p2 = parse_control_points(*control_points)

    t = tf.linspace(0., 1., num=smoothness)
    points = (1 - t) * p1 + t * p2

    print(tf.round(tf.stack(points*10, axis=1)))

    return points

def bezier_q(control_points, smoothness):
    p1, p2, p3 = parse_control_points(*control_points)

    t = tf.linspace(0., 1., num=smoothness)
    points = ((1 - t)**2) * p1 + 2 * (1 - t) * t * p2 + (t**2) * p3

    return points

def bezier_c(control_points, smoothness):
    p1, p2, p3, p4 = parse_control_points(*control_points)

    t = tf.linspace(0., 1., num=smoothness)
    points = ((1 - t)**3) * p1 + 3 * ((1 - t)**2) * t * p2 + 3 * (1 - t) * (t**2) * p3 + (t**3) * p4

    return points

def to_image(width, height, points):
    # Reverse since x are cols and y are rows
    cols, rows = tf.unstack(points)

    # We first scale the (0-1) coordinates to their actual sizes and we then
    # round to the nearest integer and finally cast to int32 in order to get
    # the index values of the coordinates
    rows, cols = tf.cast(tf.round([rows*height, cols*width]), tf.int32)

    # Pairs x and y indices into [number_of_coordinates, 2=(row,col)]
    indices = tf.stack([rows, cols], axis=1)
    updates = tf.repeat(1.0, repeats=tf.shape(indices)[0])

    # Build the image
    img = tf.zeros((height, width))
    img = tf.tensor_scatter_nd_update(img, indices=indices, updates=updates)
    print('img', img)

    plt.rcParams['axes.facecolor'] = 'black'
    plt.imshow(img, origin='lower')
    plt.show()




width = 10
height = 10

line_cp = tf.constant([(0.1,0.8), (0.8,0.1)])

bezierq_cp = tf.constant([(0.2,0.3), (0.5,0.85), (0.8,0.3)])

bezierc_cp = tf.constant([(0.2,0.3), (0.48,0.25), (0.58, 0.97), (0.8,0.3)])


c = line(line_cp, smoothness=50)
# c = bezier_q(bezierq_cp, smoothness=100)
# c = bezier_c(bezierc_cp, smoothness=100)
to_image(width, height, c)




#%%
svg_begin = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">\n'
# cp = [20,30, 48,25, 58,97, 80,30]
cp = [1,8, 8,1]
# cp = [20,30, 50,85, 80,30]
path_str = f'<path fill="none" stroke="#ffffff" d="M {cp[0]},{cp[1]} L {cp[2]},{cp[3]}" />'
# path_str = f'<path fill="none" stroke="#ffffff" d="M {cp[0]},{cp[1]} Q {cp[2]},{cp[3]} {cp[4]},{cp[5]}" />'
# path_str = f'<path fill="none" stroke="#ffffff" d="M {cp[0]},{cp[1]} C {cp[2]},{cp[3]} {cp[4]},{cp[5]} {cp[6]},{cp[7]}" />'
svg_end = '</svg>\n'

svg = ''.join([svg_begin, path_str, svg_end])
print('svg', svg)

img = cairosvg.svg2png(svg.encode('utf-8'))
img = tf.io.decode_jpeg(img, channels=1)

plt.imshow(img, origin='lower')
plt.show()

#%%
# p1x = 0.1#0.2
# p1y = 0.3#0.5

# p2x = 0.6#0.8
# p2y = 0.4#0.7

# t = np.array([0.0, 0.25, 0.5, 1.0])

# xt = (1 - t) * p1x + t * p2x
# yt = (1 - t) * p1y + t * p2y

# print('----before-----')
# print('xt\n', xt)
# print('yt\n', yt)

p1 = np.array(([[0.2], [0.1]], [[0.5], [0.3]])) # 2 x n_samples x 1
p2 = np.array(([[0.8], [0.6]], [[0.7], [0.4]]))

t = np.linspace(0, 1, num=3)
# t = np.array([0.0, 0.5, 1.0])

xt = (1 - t) * p1 + t * p2
# xt = (1 - t) * p1x + t * p2x

print('---after----')
print('xt\n', xt)
#%%

tf.repeat(3, repeats=4)