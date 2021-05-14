#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf

#%%

start = (1, 1)
end = (6, 4)
b = (7, 9)
# b = (10, 12)

def build_c(start, end, b):
    p = np.zeros((10,10,3))
    p[start] = [1., 1., 0.]
    p[end] = [1., 0.5, 0.]
    p[b] = [0., 0.9, 0.]

    def dist(p1, p2):
        dx = p1[1] - p2[1]
        dy = p1[0] - p2[0]
        return np.sqrt(dx**2 + dy**2)

    def ut_q(t):
        return ((1-t)**2) / (t**2 + (1-t)**2)


    d_start_b = dist(start, b)
    d_end_b = dist(end, b)
    t = d_start_b / (d_start_b + d_end_b)

    # print('d: start,b', d_start_b)
    # print('t:', t)

    c = ut_q(t) * np.array(start) + (1 - ut_q(t)) * np.array(end)
    c = tuple(np.round(c).astype(int))

    p[c] = [0., 0., 0.8]


    def ratio_q(t):
        return np.abs((t**2 + (1-t)**2 - 1) / (t**2 + (1-t)**2))

    a = np.array(b) + ((np.array(b) - np.array(c)) / ratio_q(t))
    a = tuple(np.round(a).astype(int))

    # p[a] = [0.7, 0., 0.]

    # print('a',a)


    # plt.imshow(p)
    # plt.show()


    def bezier_q(p1, p2, p3, smoothness):
        p1 = np.expand_dims(p1, axis=-1)
        p2 = np.expand_dims(p2, axis=-1)
        p3 = np.expand_dims(p3, axis=-1)

        t = np.linspace(0., 1., num=smoothness)
        points = ((1 - t)**2) * p1 + 2 * (1 - t) * t * p2 + (t**2) * p3

        return points


    points = bezier_q(start, a, end, smoothness=50)
    points = np.round(points).astype(int)
    points = np.clip(points, 0, 99)
    rows, cols = points

    cimg = np.zeros((10,10))
    cimg[rows,cols] = 1.

    plt.imshow(cimg)
    plt.show()

    return cimg

# build_c(start, end, b)


#%%

def bezier_q(batch_size, smoothness):
    p1, p2, p3 = tf.unstack(tf.random.uniform(shape=(3,batch_size,2,1)))
    # p1 = [[[.1], [.2]]]
    # p2 = [[[.6], [.7]]]
    # p3 = [[[.8], [.2]]]

    t = tf.linspace(0., 1., num=smoothness)
    points = ((1 - t)**2) * p1 + 2 * (1 - t) * t * p2 + (t**2) * p3

    return points


def to_image(width, height, points):
    # Reverse since x are cols and y are rows
    cols, rows = tf.unstack(points)

    # We first scale the (0-1) coordinates to their actual sizes and we then
    # round to the nearest integer and finally cast to int32 in order to get
    # the index values of the coordinates
    rows, cols = tf.cast(tf.round([rows*height, cols*width]), tf.int32)

    plt.xlim(width)
    plt.ylim(height)
    plt.plot(rows, cols, '-')
    plt.plot([20], [40], 'ro')    
    plt.show()

    # # Pairs x and y indices into [number_of_coordinates, 2=(row,col)]
    # indices = tf.stack([rows, cols], axis=1)
    # updates = tf.repeat(1.0, repeats=tf.shape(indices)[0])

    # # Build the image
    # img = tf.zeros((height, width))
    # img = tf.tensor_scatter_nd_update(img, indices=indices, updates=updates)
    # print('img', img)

    # plt.rcParams['axes.facecolor'] = 'black'
    # plt.imshow(img)
    # plt.show()




width = 10
height = 10
smoothness = 100

c = bezier_q(batch_size=1, smoothness=smoothness)

# for c_ in tf.unstack(c):
    # to_image(width, height, c_)


c = tuple(np.round(c[0]*[[height-1],[width-1]]).astype(int))
img = np.zeros((10,10))
img[c] = 1

img = img.astype(np.uint8)

plt.imshow(img)
plt.show()

contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)



ct = np.squeeze(contours)
# print(ct)
# a = np.unique(ct, return_index=True, axis=0)[1]
# a = np.sort(a)
# a = ct[a]
# print(a)
# # print(np.argsort(np.unique(ct, return_index=True, axis=0)[1], axis=0))
ct = np.transpose(ct)


# plt.xlim(0,width)
# plt.ylim(height,0)
# plt.plot(ct[0], ct[1], '-')
# plt.show()



from sklearn.neighbors import NearestNeighbors

clf = NearestNeighbors(n_neighbors=2).fit(np.transpose(ct))
G = clf.kneighbors_graph()


import networkx as nx

T = nx.from_scipy_sparse_matrix(G)

order = list(nx.dfs_preorder_nodes(T, 0))

# print('order', order)



paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(np.transpose(ct)))]

mindist = np.inf
minidx = 0

for i in range(len(np.transpose(ct))):
    p = paths[i]           # order of nodes
    ordered = np.transpose(ct)[p]    # ordered nodes
    # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
    cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
    if cost < mindist:
        mindist = cost
        minidx = i

opt_order = paths[minidx]

nctx = ct[0][opt_order]
ncty = ct[1][opt_order]




plt.xlim(0,width)
plt.ylim(height,0)
plt.plot(nctx, ncty, 'r-')
plt.show()

npoints = np.stack((nctx, ncty), axis=1)
npoints = npoints[..., ::-1]# tmp
print(npoints)


start = npoints[0]
end = npoints[-1]
b = npoints[len(npoints)//2]

print(start, end, b)

genimg = build_c(tuple(start), tuple(end), tuple(b))


genimg = tf.constant(genimg, dtype=tf.float32)
img = tf.constant(img, dtype=tf.float32)

# tf.print(img, summarize=-1)
# tf.print(genimg, summarize=-1)

loss = tf.keras.losses.MeanSquaredError()(img, genimg).numpy()
print('loss', loss)


# plt.xlim(0,width)
# plt.ylim(height,0)
# plt.plot(np.transpose(a)[0], np.transpose(a)[1], '-')
# plt.show()


# c_img = np.zeros_like(img)
# for i, contour in enumerate(contours):
#     # print('-----')
#     # print(contour.shape)
#     c_img = cv.drawContours(c_img, contours, i, np.random.randint(0, 256, size=3).tolist(), 1)



# print(np.array(contours)[0])

# plt.imshow(c_img)
# plt.show()
# print('done')

#%%
