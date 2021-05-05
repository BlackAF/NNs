#%%
import matplotlib.pyplot as plt
import cairosvg
import numpy as np
import math

#%%
def build_q(start, end, b):
    p = np.zeros((100,100,3))
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
    # points = np.clip(points, 0, 99)
    # rows, cols = points

    # cimg = np.zeros((10,10))
    # cimg[rows,cols] = 1.

    # plt.imshow(cimg)
    # plt.show()

    return c, a, points


#%%
path = np.array([(19, 81), (18, 81), (17, 80), (14, 80), (13, 79),
    (11, 79), (11, 78), (9, 78), (9, 77), (7, 77), (7, 76), (5, 76), (5, 75),
    (4, 75), (4, 74), (3, 74), (3, 73), (2, 72), (2, 71), (3, 70), (3, 68),
    (4, 68), (4, 67), (6, 67), (6, 66), (7, 66), (7, 65), (9, 65), (9, 64),
    (11, 64), (11, 60), (12, 60), (13, 59), (20, 59), (20, 60), (21, 60), (22, 61),
    (23, 61), (24, 60), (33, 60), (34, 59), (50, 59), (51, 60), (59, 60), (60, 61), (67, 61),
    (67, 60), (68, 60), (69, 59), (75, 59), (75, 60), (76, 60), (76, 62), (75, 63)])


rows, cols = np.transpose(path)



start = path[0]
end = path[-1]
# b = path[12]

for b in path[1:-1]:
    plt.xlim(0, 100)
    plt.ylim(100, 0)
    plt.plot(cols, rows)

    plt.plot(start[1], start[0], 'ro')
    plt.plot(end[1], end[0], 'go')
    plt.plot(b[1], b[0], 'yo')





    c, a, points = build_c(start, end, b)


    plt.plot(c[1], c[0], color="#54FF11", marker='o')
    plt.plot(a[1], a[0], color='#333333', marker='o')


    plt.plot(points[1], points[0], 'r-')


    plt.show()


#%%
opath = path = np.array([(19, 81), (18, 81), (17, 80), (14, 80), (13, 79),
    (11, 79), (11, 78), (9, 78), (9, 77), (7, 77), (7, 76), (5, 76), (5, 75),
    (4, 75), (4, 74), (3, 74), (3, 73), (2, 72), (2, 71), (3, 70), (3, 68),
    (4, 68), (4, 67), (6, 67), (6, 66), (7, 66), (7, 65), (9, 65), (9, 64),
    (11, 64), (11, 60), (12, 60), (13, 59), (20, 59), (20, 60), (21, 60), (22, 61),
    (23, 61), (24, 60), (33, 60), (34, 59), (50, 59), (51, 60), (59, 60), (60, 61), (67, 61),
    (67, 60), (68, 60), (69, 59), (75, 59), (75, 60), (76, 60), (76, 62), (75, 63)])

# path = path[:30]
# plt.figure(figsize=(10,10))
# plt.xlim(0, 100)
# plt.ylim(100, 0)
# plt.plot(np.transpose(path)[1], np.transpose(path)[0])
# plt.show()

# opath = path = np.array([
#  [19, 81],
#  [18, 81],
#  [17, 80],
#  [16, 80],
#  [15, 80],
#  [14, 80],
#  [13, 79],
#  [12, 79],
#  [11, 79],
#  [11, 78],
#  [10, 78],
#  [ 9, 78],
#  [ 9, 77],
#  [ 8, 77],
#  [ 7, 77],
#  [ 7, 76],
#  [ 6, 76],
#  [ 5, 76],
#  [ 5, 75],
#  [ 4, 75],
#  [ 4, 74],
#  [ 3, 74],
#  [ 3, 73],
#  [ 2, 72],
#  [ 2, 71],
#  [ 3, 70],
#  [ 3, 69],
#  [ 3, 68],
#  [ 4, 68],
#  [ 4, 67],
#  [ 5, 67],
#  [ 6, 67],
#  [ 6, 66],
#  [ 7, 66],
#  [ 7, 65],
#  [ 8, 65],
#  [ 9, 65],
#  [ 9, 64],
#  [10, 64],
#  [11, 64],
#  [11, 63],
#  [11, 62],
#  [11, 61],
#  [11, 60],
#  [12, 60],
#  [13, 59],
#  [14, 59],
#  [15, 59],
#  [16, 59],
#  [17, 59],
#  [18, 59],
#  [19, 59],
#  [20, 59],
#  [20, 60],
#  [21, 60],
#  [22, 61],
#  [23, 61],
#  [24, 60],
#  [25, 60],
#  [26, 60],
#  [27, 60],
#  [28, 60],
#  [29, 60],
#  [30, 60],
#  [31, 60],
#  [32, 60],
#  [33, 60],
#  [34, 59],
#  [35, 59],
#  [36, 59],
#  [37, 59],
#  [38, 59],
#  [39, 59],
#  [40, 59],
#  [41, 59],
#  [42, 59],
#  [43, 59],
#  [44, 59],
#  [45, 59],
#  [46, 59],
#  [47, 59],
#  [48, 59],
#  [49, 59],
#  [50, 59],
#  [51, 60],
#  [52, 60],
#  [53, 60],
#  [54, 60],
#  [55, 60],
#  [56, 60],
#  [57, 60],
#  [58, 60],
#  [59, 60],
#  [60, 61],
#  [61, 61],
#  [62, 61],
#  [63, 61],
#  [64, 61],
#  [65, 61],
#  [66, 61],
#  [67, 61],
#  [67, 60],
#  [68, 60],
#  [69, 59],
#  [70, 59],
#  [71, 59],
#  [72, 59],
#  [73, 59],
#  [74, 59],
#  [75, 59],
#  [75, 60],
#  [76, 60],
#  [76, 61],
#  [76, 62],
#  [75, 63]
#  ])

# 3=quadratic 4=cubic
order = 4


# Coffeficients for quadratic https://pomax.github.io/bezierinfo/#curvefitting
M_3 = np.array([
    [ 1, 0, 0],
    [-2, 2, 0],
    [1, -2, 1]
])
# Coefficients for cubic https://pomax.github.io/bezierinfo/#curvefitting
M_4 = np.array([
    [  1, 0, 0, 0],
    [ -3, 3, 0, 0],
    [ 3, -6, 3, 0],
    [-1, 3, -3, 1]
])

M_5 = np.array([
    [ 1,   0,   0,  0, 0],
    [-4,   4,   0,  0, 0],
    [ 6, -12,   6,  0, 0],
    [-4,  12, -12,  4, 0],
    [ 1,  -4,   6, -4, 1]
])


def comb(n, k):
    return math.factorial(n) //  (math.factorial(k) * math.factorial(n - k))

def M(n):
    n -= 1
    coef = [[comb(n, i) * comb(i, k) * (-1)**(i - k) for k in range(i + 1)] for i in range(n + 1)]
    # padding with zeros to create a square matrix
    return np.array([row + [0] * (n + 1 - len(row)) for row in coef])






def dist(p1, p2):
    dx = p1[1] - p2[1]
    dy = p1[0] - p2[0]
    return np.sqrt(dx**2 + dy**2)


def bezier(control_points, order, smoothness=50, t=None):
    if t is None:
        t = np.linspace([0.], [1.], num=smoothness)
        pws = range(order)
        t = t ** pws

    points = t @ M(order) @ control_points
    # points = ((1 - t)**3) * p1 + 3 * ((1 - t)**2) * t * p2 + 3 * (1 - t) * (t**2) * p3 + (t**3) * p4

    return points


def get_ts(path):
    Xbar = path - np.mean(path, axis=0)
    Xcov = np.transpose(Xbar) @ Xbar / len(path)
    A = np.linalg.inv(Xcov)
    V = path[1:, :] - path[:-1, :]
    t = np.diag(V @ A @ np.transpose(V)) ** (1/2)
    V2 = path[2:, :] - path[:-2, :]
    t2 = np.diag(V2 @ A @ np.transpose(V2))
    theta = np.zeros((len(path)-1, 1))

    for i in range(1, len(path)-1):
        theta[i] = min(np.math.pi - np.math.acos( (t[i-1]**2 + t[i]**2 - t2[i-1]) / (2 * t[i] * t[i-1]) ), np.math.pi/2)


    h = np.zeros((len(path)-1, 1))
    h[0] = t[0] * (1 + (1.5 * theta[1] * t[1]) / (t[0] + t[1]))


    for i in range(1, len(path)-2):
        h[i] = t[i] * (1 + (1.5 * theta[i] * t[i-1])/(t[i-1] + t[i]) + (1.5 * theta[i+1] * t[i+1])/(t[i] + t[i+1]))


    h[-1] = t[-1] * (1 + (1.5 * theta[-1] * t[-2]) / (t[-2] + t[-1]))
    h = np.insert(h, 0, [0], axis=0)
    h = np.cumsum(h, axis=0)
    h = h / h[-1]


    return h


def fit_curve(path):
    # t = [0.]

    # for i in range(1, len(path)):
    #     t.append(t[-1] + dist(path[i], path[i-1]))

    # # Scale between [0, 1]
    # t /= t[-1]

    # t = np.expand_dims(t, axis=-1)

    t = get_ts(path)

    powers = range(order)

    t = t ** powers

    bt = t @ M(order)

    p = np.linalg.pinv(bt) @ path


    # print('P', p)


    itera = 0
    resid_old = 0
    resid_new = (bt @ p) - path

    for _ in range(2):
        deriv = (order-1) * (t[:, :-1] @ M(order-1)) @ (p[1:, :] - p[:-1, :])
        new_t = t[:, 1] - ((deriv[:, 0] * resid_new[:, 0]) + (deriv[:, 1] * resid_new[:, 1])) \
                / (deriv[:, 0]**2 + deriv[:, 1]**2)
        # Scale between [0, 1]
        new_t = -np.min(new_t) + new_t
        new_t = new_t / np.max(new_t)
        # print('new_t', new_t)
        t = np.expand_dims(new_t, axis=-1) ** powers
        bt = t @ M(order)
        p = np.linalg.pinv(bt) @ path
        resid_old = resid_new
        resid_new = (bt @ p) - path
        # break
    

    # print('new p ', p)


    # C = np.linalg.inv(M(order)) @ np.linalg.inv((np.transpose(t)@t)) @ (np.transpose(t)@path)


    # print(C)
    # print(np.transpose(C))




    # cps = np.array([[10,20], [4,85], [16,67], [46,7], [50,63]])
    # ps = bezier(cps, order=5, smoothness=100)

    # print(ps)


    # # curve = bezier(C, order=order, t=t)
    curve = bezier(p, order=order, t=t)

    
    # curve = np.transpose(curve)
    # plt.figure(figsize=(10,10))
    # plt.xlim(0, 100)
    # plt.ylim(100, 0)
    # plt.plot(np.transpose(opath)[1], np.transpose(opath)[0])
    # # plt.plot(path[0, 1], path[0, 0], 'ro')
    # # plt.plot(opath[-1, 1], opath[-1, 0], 'go')
    # plt.plot(curve[1], curve[0], 'r-')
    # plt.show()
    # curve = np.transpose(curve)



    # print(len(path))
    # print(len(curve))

    # print(path[-1])
    # print(curve[-1])

    def mse(a, b):
        assert len(a) == len(b)
        return ((a - b)**2).mean()

    loss = mse(path, curve)
    print('-loss-\n', loss)


    # __L = (-2 * np.transpose(t)) @ (path - (t @ M(order) @ C))
    # __L = np.mean(__L)
    # print('__L\n', "{:10.17f}".format(__L))
    return curve, loss, p



# c = fit_curve(path[:30])
# c = fit_curve(path[29:39])
# c = fit_curve(path[38:49])
# c = fit_curve(path[0:38])


# plt.figure(figsize=(10,10))
# plt.xlim(0, 100)
# plt.ylim(100, 0)
# plt.plot(np.transpose(path)[1], np.transpose(path)[0])
# plt.show()


max_error = 0.0905

start = 0
end = len(path)
min_points = 3

new_path = []

while len(path) - start > min_points:
    curve, loss, control_points = fit_curve(path[start:end])

    if loss < max_error:
        print('start: ', start, 'end: ', end-1)
        curve = np.transpose(curve)

        plt.figure(figsize=(10,10))
        plt.xlim(0, 100)
        plt.ylim(100, 0)
        plt.plot(np.transpose(opath)[1], np.transpose(opath)[0])
        plt.plot(opath[start, 1], opath[start, 0], 'ro', markersize=2)
        plt.plot(opath[-1, 1], opath[-1, 0], 'go')
        plt.plot(curve[1], curve[0])
        plt.show()

        new_path.append(control_points)
        # print('control_points', control_points)

        start = end - 1
        end = len(path)

        continue

    if end - start <= min_points:
        print('start: ', end, 'end: ', end)
        
        start = end - 1
        end = len(path)

        new_path.append(control_points)
        
        continue

    end -= 1

print('new_path')
print(new_path)

if start != len(path) - 1:
    print('remaining:', path[start:])
# if start != len(path)-1 add remaining points as line path

new_path = np.array(new_path)

for i in range(len(new_path)-1):
    # Join start and end point in the middle
    end = new_path[i, -1]
    start = new_path[i+1, 0]
    # Find middle point
    join = end + ((start - end) / 2)
    # Update end point of this curve
    new_path[i, -1] = join
    # Update start point of the next curve
    new_path[i+1, 0] = join

#     # k = dist(new_path[i+1, 1], join) / dist(join, new_path[i, -2])
#     # new_cp = join + k * (join - new_path[i, -2])
#     # new_path[i+1, 1] = new_cp
#     # print('k', k)

#     # last_control_point = new_path[i+1, 1]
#     # a = math.atan2(last_control_point[0] - join[0], last_control_point[1] - join[1])
#     # d = dist(new_path[i, -2], join)
#     # new_control_point_x = join[1] + d * math.cos(a)
#     # new_control_point_y = join[0] + d * math.sin(a)

#     # new_path[i, -2] = (new_control_point_y, new_control_point_x)


# # print(new_path)


plt.figure(figsize=(10,10))
for cps in new_path:
    curve = bezier(cps, order=order)
    curve = np.transpose(curve)

    plt.xlim(0, 100)
    plt.ylim(100, 0)
    # plt.plot(np.transpose(opath)[1], np.transpose(opath)[0], 'b-')
    # plt.plot(opath[start, 1], opath[start, 0], 'ro', markersize=2)
    # plt.plot(opath[-1, 1], opath[-1, 0], 'go')
    plt.plot(curve[1], curve[0], 'r-')
    # plt.plot([cps[-1, 1], cps[-2, 1]], [cps[-1, 0], cps[-2, 0]], 'b-')
    # plt.plot([cps[1, 1], cps[0, 1]], [cps[1, 0], cps[0, 0]], 'r-')
plt.show()





# s = f'M {new_path[0][0,1]},{new_path[0][0, 0]}'

# for p in new_path:
#     p1, p2, end = p[1:]
#     s += f' C {p1[1]},{p1[0]} {p2[1]},{p2[0]} {end[1]},{end[0]}'
    
# print(s)

# s = f'\t<path stroke="#ff0000" fill="none" d="{s}" />\n'

# svg_begin = f'<svg xmlns="http://www.w3.org/2000/svg" width="{100}" height="{100}">\n'
# svg_end = '</svg>\n'
# svg = ''.join([svg_begin, s, svg_end])


# print(svg)

# with open('path.svg', 'w') as f:
    # f.write(svg)




#%%

#%%

#%%
import tensorflow as tf
import math

opath = path = np.array([(19, 81), (18, 81), (17, 80), (14, 80), (13, 79),
    (11, 79), (11, 78), (9, 78), (9, 77), (7, 77), (7, 76), (5, 76), (5, 75),
    (4, 75), (4, 74), (3, 74), (3, 73), (2, 72), (2, 71), (3, 70), (3, 68),
    (4, 68), (4, 67), (6, 67), (6, 66), (7, 66), (7, 65), (9, 65), (9, 64),
    (11, 64), (11, 60), (12, 60), (13, 59), (20, 59), (20, 60), (21, 60), (22, 61),
    (23, 61), (24, 60), (33, 60), (34, 59), (50, 59), (51, 60), (59, 60), (60, 61), (67, 61),
    (67, 60), (68, 60), (69, 59), (75, 59), (75, 60), (76, 60), (76, 62), (75, 63)])

# path = path[:10]

def dist(p1, p2):
    dx = p1[1] - p2[1]
    dy = p1[0] - p2[0]
    return np.sqrt(dx**2 + dy**2)

def comb(n, k):
    return math.factorial(n) //  (math.factorial(k) * math.factorial(n - k))

def M(n):
    n -= 1
    coef = [[comb(n, i) * comb(i, k) * (-1)**(i - k) for k in range(i + 1)] for i in range(n + 1)]
    # padding with zeros to create a square matrix
    return np.array([row + [0] * (n + 1 - len(row)) for row in coef])

t = [0.]
for i in range(1, len(path)):
    t.append(t[-1] + dist(path[i], path[i-1]))

# Scale between [0, 1]
t /= t[-1]
t = np.expand_dims(t, axis=-1)
order = 20
powers = range(order)
t = t ** powers


__C = np.linalg.inv(M(order)) @ np.linalg.inv((np.transpose(t)@t)) @ (np.transpose(t)@path)
# print('__C\n', __C)



t = tf.constant(t, dtype=tf.float32)
m = tf.constant(M(order), dtype=tf.float32)
p = tf.constant(path, dtype=tf.float32)/100.
C = tf.Variable(tf.random.normal((order, 2)), dtype=tf.float32)


# with tf.GradientTape() as tape:
#     e = tf.transpose((p - (t @ m @ C))) @ (p - (t @ m @ C))

# grads = tape.gradient(e, C)
# print(grads)


opt = tf.keras.optimizers.Adam(learning_rate=0.1)
l = tf.keras.losses.MeanSquaredError()

def f():
    # e = tf.transpose((p - (t @ m @ C))) @ (p - (t @ m @ C))
    e = l(p, (t @ m @ C))
    # e = tf.keras.losses.mse(p, (t @ m @ C))
    # e = tf.math.reduce_mean(e)
    print(e)
    return e


for _ in range(10000):
    opt.minimize(f, var_list=[C])
# print(__C)
C = np.array(C) * 100
# print(C)



curve = np.transpose(t @ m @ C)


plt.xlim(0, 100)
plt.ylim(100, 0)
plt.plot(np.transpose(opath)[1], np.transpose(opath)[0])
plt.plot(opath[0, 1], opath[0, 0], 'ro', markersize=2)
plt.plot(opath[-1, 1], opath[-1, 0], 'go')
plt.plot(curve[1], curve[0])
plt.show()
