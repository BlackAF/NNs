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




def fit_curve(path):
    t = [0.]

    for i in range(1, len(path)):
        t.append(t[-1] + dist(path[i], path[i-1]))

    # Scale between [0, 1]
    t /= t[-1]

    t = np.expand_dims(t, axis=-1)

    # 3=quadratic 4=cubic
    order = 4
    powers = range(order)

    t = t ** powers

    # print(t)

    C = np.linalg.inv(M(order)) @ np.linalg.inv((np.transpose(t)@t)) @ (np.transpose(t)@path)


    # print(C)
    # print(np.transpose(C))



    def bezier(control_points, order, smoothness=50, t=None):
        if t is None:
            t = np.linspace([0.], [1.], num=smoothness)
            pws = range(order)
            t = t ** pws

        points = t @ M(order) @ control_points
        # points = ((1 - t)**3) * p1 + 3 * ((1 - t)**2) * t * p2 + 3 * (1 - t) * (t**2) * p3 + (t**3) * p4

        return points


    # cps = np.array([[10,20], [4,85], [16,67], [46,7], [50,63]])
    # ps = bezier(cps, order=5, smoothness=100)

    # print(ps)


    curve = bezier(C, order=order, t=t)
    # curve = np.transpose(curve)

    # plt.xlim(0, 100)
    # plt.ylim(100, 0)
    # plt.plot(np.transpose(opath)[1], np.transpose(opath)[0])
    # plt.plot(path[0, 1], path[0, 0], 'ro')
    # plt.plot(opath[-1, 1], opath[-1, 0], 'go')
    # plt.plot(curve[1], curve[0])
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

    return curve, loss, C

max_error = 0.1

start = 0
end = len(path) - 1

while len(path) - start > 4:
    if end - start <= 4 - 1:
        print('start: ', end, 'end: ', end)
        start = end
        end = len(path) - 1
        continue

    curve, loss, control_points = fit_curve(path[start:end+1])

    if loss < max_error:
        print('start: ', start, 'end: ', end)
        curve = np.transpose(curve)

        plt.xlim(0, 100)
        plt.ylim(100, 0)
        plt.plot(np.transpose(opath)[1], np.transpose(opath)[0])
        plt.plot(opath[start, 1], opath[start, 0], 'ro', markersize=2)
        plt.plot(opath[-1, 1], opath[-1, 0], 'go')
        plt.plot(curve[1], curve[0])
        plt.show()

        print('control_points', control_points)

        start = end
        end = len(path) - 1

        continue

    end -= 1


