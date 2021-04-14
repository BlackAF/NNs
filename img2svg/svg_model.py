#%%
import tensorflow as tf
import cairosvg
import cairocffi

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, MaxPooling2D, Conv2D, GlobalAveragePooling2D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset

tf.get_logger().setLevel('ERROR')
tf.config.list_physical_devices('GPU')

#%%
class IMG2SVG(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.svg_model = self.load_svg_model()

    def load_svg_model(self):
        svg_model = Sequential()

        svg_model.add(Input((10, 10, 1)))
        
        svg_model.add(Conv2D(128, 3, activation='relu', padding='same'))
        svg_model.add(Conv2D(128, 3, activation='relu', padding='same'))
        svg_model.add(MaxPooling2D(strides=2))

        svg_model.add(Conv2D(64, 3, activation='relu', padding='same'))
        svg_model.add(Conv2D(64, 3, activation='relu', padding='same'))
        svg_model.add(MaxPooling2D(strides=2))

        svg_model.add(GlobalAveragePooling2D())
        svg_model.add(Dense(32, activation='relu'))
        svg_model.add(Dense(4, activation='sigmoid'))
        svg_model.add(Reshape((-1, 2)))

        svg_model.compile()

        return svg_model

    def svg_to_img(self, control_points):
        # svg_begin = tf.constant('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">\n')
        # path = tf.strings.format('\t<path fill="none" stroke="#00FF00" d="M{},{} L{},{}" />\n', control_points)
        # svg_end = tf.constant('</svg>\n')
        
        # svg = tf.strings.join([svg_begin, path, svg_end])

        # def parse_svg(svgs):
        #     imgs = []
        #     for svg in svgs:
        #         img = cairosvg.svg2png(bytestring=svg.numpy())
        #         img = tf.io.decode_png(img, channels=3)
        #         img = tf.cast(img, tf.float32)
        #         img /= 255.0

        #         imgs.append(img)

        #     return tf.stack(imgs)

        # imgs = tf.py_function(func=parse_svg, inp=[[svg]], Tout=tf.float32)


        def tes(svg):
            return tf.ones((1, 10, 10, 1), dtype=tf.float32)

        t = tf.py_function(func=tes, inp=[control_points], Tout=tf.float32)

        return t

    @tf.function
    def train_step(self, data):
        width = data.shape[2]
        height = data.shape[1]
        batch_size = data.shape[0]

        # This will returns the coordinates of each pixel that isn't 0
        # in this format [(batch_index, row, col),]
        data_points = tf.where(data[:, :, :, 0])
        xy_coords = data_points[:, 1:]
        batch_row_ids = data_points[:, 0]
        # Group the above result by batch: [batch, number_of_coords, (row,col)]
        data_points = tf.RaggedTensor.from_value_rowids(xy_coords, batch_row_ids, nrows=batch_size)
        data_points = data_points.to_tensor()
        # data_points /= [height, width]
        # dp = data_points / [height, width]
        # tf.print('DP\n', data_points)
        # tf.print('DP\n', dp)
        # # Get the 1d index positions of the coordinates
        # data_points = data_points[:, :, 0] * width + data_points[:, :, 1]
        # # Scale between 0 and 1
        # data_points = data_points / ((width * height) - 1)
        # # This will 0 pad to the max length
        # data_points = data_points.to_tensor()
        # # tf.print('data_points\n', tf.round(data_points*99.))

        with tf.GradientTape() as tape:
            svg = self.svg_model(data)
            # tf.print(svg, summarize=-1)
            # tf.print('\nsvg\n', svg)
            svg = tf.expand_dims(svg, axis=-1) # [batch, n_control_points, 2=[row,col], 1]
            p1, p2 = tf.unstack(svg, axis=1) # [ p1=[[batch, 2, 1], batch2..], p2=[[batch1, 2, 1], batch2..], p3.. ]


            t = tf.linspace(0., 1., num=5)
            points = (1 - t) * p1 + t * p2 # [batch, 2=[rows,cols], n_points]

            #--- working points method ---#
            # Go from [rows, cols] to [(x, y),]
            # TODO: transpose is costly, maybe keep it in [rows,cols] and format data instead
            points = tf.transpose(points, perm=[0, 2, 1]) # [batch, n_points, 2=[row,col]]
            # Scale points
            points *= [height, width]
            x_rounded_NOT_differentiable = tf.round(points)
            # tf.print('bef points\n', points, summarize=-1)
            points = (points - (tf.stop_gradient(points) - x_rounded_NOT_differentiable))
            # tf.print('aft points\n', points, summarize=-1)
            #--- working points method ---#

                        
            pos = points[:, :, 0] * width + points[:, :, 1]
            pos = tf.argsort(pos, axis=1)

            points = tf.gather(points, pos, batch_dims=1)

            d = points[:, 1:] - points[:, :-1]
            d = tf.math.abs(d)
            d = tf.math.reduce_sum(d, axis=2)
            d = tf.pad(d, paddings=[[0, 0], [1, 0]], constant_values=1.)

            w = tf.where(d)
            w = tf.RaggedTensor.from_value_rowids(w[:, 1], w[:, 0])

            points = tf.RaggedTensor.from_tensor(points)
            points = tf.gather(points, w, axis=1, batch_dims=1)
            points = points.to_tensor(-1)

            pad = 5 - tf.shape(points)[1]
            points = tf.pad(points, paddings=[[0, 0], [0, pad], [0, 0]], constant_values=-1)

            tf.print('aft points\n', points, summarize=-1)

            # a = tf.gather_nd(a, tf.expand_dims(c.values, axis=-1))

            # ga = tf.gather(points, [0, 2, 1, 3, 4], axis=1)
            # tf.print('gather\n', ga, summarize=-1)

            # Insert a new dimension for rows and cols to get [batch, num_points, row, col, 2]
            # points = points[:, :, tf.newaxis, tf.newaxis, :]
            # tf.print(points)

            # y = points
            # y = tf.tile(points, [1, 1, height, width, 1])

            # rows = tf.range(tf.cast(height, dtype=tf.float32))
            # cols = tf.range(tf.cast(width, dtype=tf.float32))

            # grid = tf.meshgrid(rows, cols, indexing='ij')
            # grid = tf.stack(grid, axis=-1)

            # y = y - grid
            # y = tf.math.abs(y)

            # y = 1. - (y * (1. / (y+1e-07)))
            # y = (y - (tf.stop_gradient(y) - tf.round(y)))

            # y = y[:, :, :, :, 0] * y[:, :, :, :, 1]

            # y = 1. - y

            # y = tf.math.reduce_prod(y, axis=1)
            # y = 1. - y

            # y = y[:, :, :, :, 0] + y[:, :, :, :, 1]
            # y = tf.clip_by_value(y, 0., 1.) # DIFFERENTIABLE BUT GRADIENTS ALL GO TO ZERO SO USELESS AF


            # y = y[:, 0, :, :]
            # tf.print('--y--')
            # tf.print(y, y.shape, summarize=-1)



            # tf.print('PTS\n', points)
            # Get the 1d index positions of the coordinates
            # points = points[:, :, 0] * width + points[:, :, 1]
            # Scale points
            # points = points / ((width * height) - 1)
            # TODO: remove since this will be a ragged tensor so i can just .to_tensor(0)
            # points = tf.pad(points, paddings=[[0, 0], [0, tf.shape(data_points)[1]-tf.shape(points)[1]]])
            # tf.print('points\n', points, summarize=-1)

            # data_points1d = tf.expand_dims(data_points1d, axis=0)
            # tf.print('data points1s\n', data_points1d, summarize=-1)

            # points_sum = tf.math.reduce_sum(points1d)
            # tf.print('points sum\n', points_sum, summarize=-1)

            # data_points_sum = tf.math.reduce_sum(data_points1d)
            # tf.print('data points sum\n', data_points_sum, summarize=-1)

            # tf.print('ori points\n', points, summarize=-1)
            # Reverse since x are cols and y are rows
            # points = tf.reverse(points, axis=[0])
            # tf.print('aft points\n', points, summarize=-1)
            # points = tf.transpose(points)
            # tf.print('transpose points\n', points, summarize=-1)


            # data_points = tf.cast(tf.where(data[0, :, :, 0]), dtype=tf.float32)
            # tf.print('data points\n', data_points, summarize=-1)
            
            # Reverse since x are cols and y are rows
            # cols, rows = tf.unstack(points)

            # We first scale the (0-1) coordinates to their actual sizes and we then
            # round to the nearest integer and finally cast to int32 in order to get
            # the index values of the coordinates
            # rows = tf.cast(tf.round(rows * C.HEIGHT), tf.int32)
            # cols = tf.cast(tf.round(cols * C.WIDTH), tf.int32)
            # rows_c = tf.cast(tf.round(rows * C.HEIGHT), tf.int32)
            # cols_c = tf.cast(tf.round(cols * C.WIDTH), tf.int32)
            # tf.print('rows\n', rows)

            # for _ in cols:
                # tf.print(_)

            # Pairs x and y indices into [number_of_coordinates, 2=(row,col)]
            # indices = tf.stack([rows_c, cols_c], axis=1)
            # updates = tf.repeat(1.0, repeats=tf.shape(indices)[0])

            # Build the image
            # img = tf.zeros((C.HEIGHT, C.WIDTH))
            # img = tf.tensor_scatter_nd_update(img, indices=indices, updates=updates)
            # img = tf.expand_dims(img, axis=-1)

            # tf.print('img\n', img, summarize=-1)
            # tf.print(img.shape)
            # tf.print(data[0, :, :, 0], summarize=-1)
            # tf.print('stcal', tf.stack([rows, cols]), summarize=-1)
            # tf.print('data_points\n', tf.round(data_points*99.), 'points\n', tf.round(points*99.), summarize=-1)
            # tf.print(tf.math.reduce_sum(data_points, axis=1))
            # tf.print('g tr')
            # tf.print(data.shape, summarize=-1)
            # tf.print()
            # y = tf.expand_dims(y, axis=-1)
            # tf.print('y')
            # tf.print(y[0, :, :, 0], summarize=-1)
            # w = tf.where(y[0, :, :, 0])
            # w = tf.expand_dims(w, 0)
            # w = tf.tile(w, [1, 5, 1])
            # tf.print(w.dtype, summarize=-1)
            # loss = self.compiled_loss(data, y)
            loss = self.compiled_loss(data_points, points)
            # loss = self.compiled_loss(data[0, 0:2, :, 0], tf.stack([rows, cols]))
            # tf.print('loss', loss)
        
        # tf.print([var.name for var in tape.watched_variables()])
        gradients = tape.gradient(loss, self.svg_model.trainable_variables)
        # tf.print('grads\n', gradients[10], summarize=-1)
        self.optimizer.apply_gradients(zip(gradients, self.svg_model.trainable_variables))

        return { 'loss': loss }

# tm = TrainManager()

img2svg = IMG2SVG()
img2svg.compile(Adam(lr=C.INIT_LR), loss='mse')

inp = tf.constant([
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
inp = inp[tf.newaxis, :, :, tf.newaxis]
x = Dataset.from_tensor_slices(inp)
# x = Dataset.from_tensor_slices(tf.ones((1,10,10,1)))
x = x.batch(1)

img2svg.fit(x, epochs=50)


#%%
# a = tf.round(img2svg.svg_model(inp)[0] * 10)
a = img2svg.svg_model(inp)[0]
print(tf.round(a*10))
a = tf.expand_dims(a, axis=-1)

t = tf.linspace(0., 1., num=5)
points = (1 - t) * a[0] + t * a[1]
points = tf.transpose(points)

# print(points)
print(tf.round(points*10))

#%%
class C:
    INIT_LR = 1e-03
    WIDTH = 10
    HEIGHT = 10

#%%

s = b'''<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
\t<path fill="none" stroke="#00FF00" d="M 10,12 L84,27" />
</svg>\n'''

i = cairosvg.svg2png(bytestring=s)
# i = cairosvg.svg2png(bytestring=s.encode('utf-8'))


print(s)


#%%
a = tf.constant([
    [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    ],
    [
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 1],
    ]
], dtype=tf.float32)
print(a)

ps = tf.where(a)
print()
print(ps)

width = a.shape[2]
height = a.shape[1]

ps = tf.RaggedTensor.from_value_rowids(ps[:, 1:], ps[:, 0], nrows=a.shape[0])

print()
print(ps)


# ps /= [height, width]
# print()
# print(ps)

pos = ps[:, :, 0] * width + ps[:, :, 1]

print()
print(pos)
print(pos.to_tensor())
# print(pos/15)


#%%

r = tf.constant([
    [ [2., 3.], [2., 8.] ],
    [ [1., 4.], [7., 4.] ]
])
r = tf.expand_dims(r, axis=-1)
p1, p2 = tf.unstack(r, axis=1)

# p1 = tf.constant([
#     [ [2.], [3.] ],
#     [ [1.], [4.] ]
# ])
# p2 = tf.constant([
#     [ [2.], [8.] ],
#     [ [7.], [4.] ]
# ])

p1 = tf.zeros((10, 10)).numpy()
p1[2, 1] = 1.
p1 = tf.constant(p1)
print(p1)

p2 = tf.zeros((10, 10)).numpy()
p2[2, 7] = 1.
p2 = tf.constant(p2)
print(p2)

ts = tf.linspace([0.], [1.], num=5, axis=1)
ts = tf.expand_dims(ts, axis=-1)
print('ts', ts)
# ts = tf.tile(ts, [10, 10, 1])
# print('ts', ts)
ps = (1 - ts) * p1 + ts * p2


print()
print(ps)

ps = tf.transpose(ps, perm=[0, 2, 1])

print()
print(ps)




#%%

# ---- 1D -----

y = tf.constant([[3.], [1.]])
y = tf.tile(y, [1, 5])

print()
print(y)

r = tf.range(5.)

print()
print(r)

y = y - r
y = tf.math.abs(y)

print()
print(y)

y = 1 - (y * (1 / (y+1e-07)))
y = tf.round(y)

print()
print(y)



#%%

# ---- 2D -----
y = tf.constant([
    [
        [
            [[2., 3.]]
        ],
        [
            [[1., 4.]]
        ] # num points
    ] # batch
])
y = tf.tile(y, [1, 1, 3, 5, 1])

print()
print(y)

rows = tf.range(3.)
cols = tf.range(5.)

X, Y = tf.meshgrid(rows, cols, indexing='ij')
# # # print(X)
# # # print()
# # # print(Y)

r = tf.stack([X, Y], axis=-1)

print()
print(r)

y = y - r
y = tf.math.abs(y)

print()
print(y)

# y = 1 - (y * (1 / (y+1e-07)))
# y = tf.round(y)


y = y[:, :, :, :, 0] + y[:, :, :, :, 1]

print()
print(y)

# y = y[:, :, :, :, 0] * y[:, :, :, :, 1]

# print()
# print(y)

# y = 1 - y

# print()
# print(y)


# y = tf.math.reduce_prod(y, axis=1)
# y = 1 - y

# print()
# print(y)




#%%
a = tf.constant([
    [2., 4.],
    [1., 0.],
    [4., 2.],
    [2., 1.],
    [1., 0.],
    [9., 6.],
    [2., 14.],
    [3., 1.]
])
print('a')
print(a)

# a = tf.transpose(a)
# print(a)

# print(tf.floor(log10(a[:, 1])+1))

# ta = a[:, 0] + (a[:, 1] / 10**tf.floor(log10(a[:, 1])+1) )
# print('added')
# print(ta)

# a = tf.sort(a[:, 0], axis=0)
i = tf.argsort(ta, axis=0)

print()
# print(i)

# a = tf.gather(a, i)
# a = tf.math.top_k(a, k=2)

a = tf.raw_ops.UniqueV2(x=a, axix=[1])

# b = tf.roll(a, shift=1, axis=0)[1:]
# b = tf.pad(b, paddings=[[1, 0], [0, 0]])

# r = tf.range(a.shape[0])

print('a')
print(a)


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    numerator = tf.where(tf.math.is_inf(numerator), tf.zeros_like(numerator), numerator)
    return numerator / denominator

# ab = a - b
# print('ab')
# print(ab)


#%%

# ---- UNIQUE 1D ----

a = tf.constant([2., 7., 4., 7., 2.])

print()
print(a)

a = tf.sort(a, axis=0)

print()
print(a)

d = a[1:] - a[:-1]

print()
print(d)

w = tf.where(d) + 1
w = tf.pad(w, [[1, 0], [0, 0]])

print()
print(w)

a = tf.gather_nd(a, w)

print()
print(a)


#%%

# ---- UNIQUE 2D ----

a = tf.constant([
    [
        [2., 4.],
        [1., 0.],
        [4., 2.],
        [9., 6.],
        [2., 1.],
        [1., 0.],
        [9., 6.],
        [2., 14.],
        [3., 1.],
        [0., 5.],
    ],
    [
        [2., 3.],
        [2., 3.],
        [2., 3.],
        [2., 3.],
        [2., 3.],
        [2., 3.],
        [2., 3.],
        [2., 3.],
        [2., 3.],
        [2., 3.],
    ],
    [
        [2., 4.],
        [9., 7.],
        [4., 2.],
        [9., 7.],
        [2., 5.],
        [6., 7.],
        [8., 6.],
        [2., 12.],
        [0., 3.],
        [5., 3.],
    ]
])



width = 15
pos = a[:, :, 0] * width + a[:, :, 1]

print()
print(pos)

pos = tf.argsort(pos, axis=1)

print()
print(pos)

a = tf.gather(a, pos, batch_dims=1)

print()
print(a)

d = a[:, 1:] - a[:, :-1]
d = tf.math.abs(d)

print()
print(d)

d = tf.math.reduce_sum(d, axis=2)
d = tf.pad(d, paddings=[[0, 0], [1, 0]], constant_values=1.)

print()
print(d)

w = tf.where(d)

print()
print(w)

w = tf.RaggedTensor.from_value_rowids(w[:, 1], w[:, 0])

print()
print(w)


print(a)
a = tf.RaggedTensor.from_tensor(a)
a = tf.gather(a, w, axis=1, batch_dims=1)

print()
print(a.to_tensor(-99))





# c = tf.sparse.SparseTensor(tf.cast(a, tf.int64), tf.range(10), dense_shape=[1,1])
# c = tf.sparse.reorder(c)
# print(c)

# a = tf.gather_nd(a, tf.expand_dims(c.values, axis=-1))

# print()
# print(a)


# d = a[1:, :] - a[:-1, :]

# print()
# print(d)

