#%%
import numpy as np
import matplotlib.pyplot as plt
import time

from collections import deque
from PIL import Image

np.set_printoptions(threshold=sys.maxsize)


#%%
def outside(point):
    return point[0] < 0 or point[0] >= 100 or point[1] < 0 or point[1] >= 100

#%%
img = Image.open('opencv/skate_canny.jpg')
img = np.asarray(img)
img = np.where(img < 126, 0, 255)

print('Starting!')
start_time = time.time()

points = np.where(img)
points = np.transpose(points)

e = img #np.zeros_like(img)
e = np.expand_dims(e, axis=-1)
e = np.tile(e, [1, 1, 3])

ori_p = curr_p = (35, 83)

e[curr_p] = [255, 0, 0]
# print(points)

min_p = None
max_p = None


for _ in range(200):
    up = (curr_p[0]-1, curr_p[1])

    if outside(up):
        min_p = curr_p
        break

    found = any(np.equal(points, up).all(1))
    if found:
        np.delete(points, up, axis=0)
    else:
        min_p = curr_p
        break

    curr_p = up
    e[curr_p] = [255, 0, 0]

    # plt.figure(figsize=(10,10))
    # plt.imshow(e)
    # plt.pause(0.000005)

curr_p =  ori_p

for _ in range(200):
    down = (curr_p[0]+1, curr_p[1])

    if outside(down):
        max_p = curr_p
        break

    found = any(np.equal(points, down).all(1))
    if found:
        np.delete(points, down, axis=0)
    else:
        max_p = curr_p
        break

    curr_p = down
    e[curr_p] = [255, 0, 0]

    # plt.figure(figsize=(10,10))
    # plt.imshow(e)
    # plt.pause(0.000005)


e[min_p] = [0, 255, 0]
e[max_p] = [0, 0, 200]
plt.figure(figsize=(10,10))
plt.imshow(e)

plt.show()

print(img.shape)




end_time = time.time() - start_time
print('Done! - Took: ', end_time, 's')

#%%
img = Image.open('opencv/skate_canny.jpg')
img = np.asarray(img)
img = np.where(img < 126, 0, 255)

print('Starting!')
start_time = time.time()

points = np.where(img)
points = np.transpose(points)

# print(points)

e = img #np.zeros_like(img)
e = np.expand_dims(e, axis=-1)
e = np.tile(e, [1, 1, 3])

cols = set([col for col in points[:, 1]])
print(cols)


for col in cols:
    # e[:, col] = [255, 0, 0]

    frow = np.equal(points[:, 1], col)
    frow = np.sort(points[frow], axis=0)

    # print(frow)

    d = frow[1:] - frow[:-1]
    d = np.sum(d, axis=1)
    d = d - 1
    d = np.minimum(d, 1)
    d = np.insert(d, 0, 1, axis=0)
    d = np.nonzero(d)

    start_ps = frow[d]
    start_ps = np.transpose(start_ps)

    # print(start_ps)

    e[tuple(start_ps)] = [255, 0, 0]


    t = np.abs(frow[:-1] - frow[1:])
    t = np.sum(t, axis=1)
    t = t - 1
    t = np.minimum(t, 1)
    t = np.append(t, [1], axis=0)
    t = np.nonzero(t)

    end_ps = frow[t]
    end_ps = np.transpose(end_ps)

    # print(end_ps)

    e[tuple(end_ps)] = [0, 255, 0]

    plt.figure(figsize=(10,10))
    plt.imshow(e)


plt.show()

end_time = time.time() - start_time
print('Done! - Took: ', end_time, 's')



#%%
img = Image.open('opencv/skate_canny.jpg')
img = np.asarray(img)
img = np.where(img < 126, 0, 1)


print('Starting!')
start_time = time.time()

pimg = np.pad(img, [[1, 1], [1, 1]])


ver = pimg[1:, :] - pimg[:-1, :]
hor = pimg[:, 1:] - pimg[:, :-1]

sver = np.equal(ver, 1)[:-1, 1:-1]
ever = np.equal(ver, -1)[1:, 1:-1]

shor = np.equal(hor, 1)[1:-1, :-1]
ehor = np.equal(hor, -1)[1:-1, 1:]


sver_sorted = np.transpose(np.nonzero(sver))
height = 100
# sort by column instead of by row
sver_sorted = sver_sorted[:, ::-1]
pos = sver_sorted[:, 0] * height + sver_sorted[:, 1]
sver_sorted = sver_sorted[np.argsort(pos, axis=0)]
# put back rows to cols and cols to rows
sver_sorted = sver_sorted[:, ::-1]
# print(sver_sorted)


ever_sorted = np.transpose(np.nonzero(ever))
height = 100
# sort by column instead of by row
ever_sorted = ever_sorted[:, ::-1]
pos = ever_sorted[:, 0] * height + ever_sorted[:, 1]
ever_sorted = ever_sorted[np.argsort(pos, axis=0)]
# put back rows to cols and cols to rows
ever_sorted = ever_sorted[:, ::-1]
# print('----')
# print(ever_sorted)


cps_ver = np.stack((sver_sorted, ever_sorted), axis=1)

# print(cps_ver)

# already sorted
shor_sorted = np.transpose(np.nonzero(shor))
ehor_sorted = np.transpose(np.nonzero(ehor))
# print('----')
# print(ehor_sorted)

cps_hor = np.stack((shor_sorted, ehor_sorted), axis=1)
# print(cps_hor)


# find single points
ver_single = cps_ver[:, 0, :] - cps_ver[:, 1, :]
ver_single = np.abs(ver_single)
ver_single = np.minimum(ver_single, 1)
ver_single = ver_single[:, 0] + ver_single[:, 1]
# keep track of points that are not single
ver_not_single = np.flatnonzero(ver_single)
ver_single = 1 - ver_single
ver_single = np.flatnonzero(ver_single)
ver_single = cps_ver[ver_single]
# both start and end is the same so keep one
ver_single = ver_single[:, 0, :]

hor_single = cps_hor[:, 0, :] - cps_hor[:, 1, :]
hor_single = np.abs(hor_single)
hor_single = np.minimum(hor_single, 1)
hor_single = hor_single[:, 0] + hor_single[:, 1]
# keep track of points that are not single
hor_not_single = np.flatnonzero(hor_single)
hor_single = 1 - hor_single
hor_single = np.flatnonzero(hor_single)
hor_single = cps_hor[hor_single]
# both start and end is the same so keep one
hor_single = hor_single[:, 0, :]

# print(hor_single.shape, cps_hor.shape)
# print(ver_single.shape, cps_ver.shape)

# remove single points from main
cps_ver = cps_ver[ver_not_single]
cps_hor = cps_hor[hor_not_single]


# only keep necessary single points
singles = np.concatenate((ver_single, hor_single), axis=0)
width = 100
pos = singles[:, 0] * width + singles[:, 1]
singles = singles[np.argsort(pos, axis=0)]
idx = singles[1:] - singles[:-1]
idx = np.abs(idx)
idx = idx[:, 0] + idx[:, 1]
idx = np.minimum(idx, 1)
idx = 1 - idx
idx = np.flatnonzero(idx) + 1
singles = singles[idx]

# print(singles)




gen = np.zeros_like(img)
# gen = img
gen = np.expand_dims(gen, axis=-1)
gen = np.tile(gen, [1, 1, 3])
gen = gen.astype(np.float32)



# join paths
verstarts = {tuple(start): tuple(end) for start, end in cps_ver}
verends = {tuple(end): tuple(start) for start, end in cps_ver}
horstarts = {tuple(start): tuple(end) for start, end in cps_hor}
horends = {tuple(end): tuple(start) for start, end in cps_hor}


# print(verstarts)


def add_connected_lines(connected_point, path, add_to_front, find_horizontal):
    found_extremity = False

    while not found_extremity:
        if find_horizontal:
            end = horstarts.pop(connected_point, None)
            if not end:
                start = horends.pop(connected_point, None)

            if end:
                path.appendleft(end) if add_to_front else path.append(end)
                horends.pop(end)
                connected_point = end
                find_horizontal = False
            elif start:
                path.appendleft(start) if add_to_front else path.append(start)
                horstarts.pop(start)
                connected_point = start
                find_horizontal = False
            else:
                found_extremity = True
        
        else:
            end = verstarts.pop(connected_point, None)
            if not end:
                start = verends.pop(connected_point, None)

            if end:
                path.appendleft(end) if add_to_front else path.append(end)
                verends.pop(end)
                connected_point = end
                find_horizontal = True
            elif start:
                path.appendleft(start) if add_to_front else path.append(start)
                verstarts.pop(start)
                connected_point = start
                find_horizontal = True
            else:
                found_extremity = True


paths = []

while len(verstarts):
    path = deque()

    start = list(verstarts.keys())[0]
    end = verstarts.pop(start, None)
    verends.pop(end)
    path.append(start)
    path.append(end)
    
    add_connected_lines(connected_point=start, path=path, add_to_front=True, find_horizontal=True)
    add_connected_lines(connected_point=end, path=path, add_to_front=False, find_horizontal=True)

    paths.append(path)
    # print('path', path)

    # plt.xlim(0, 100)
    # plt.ylim(100, 0)
    # rows, cols = np.transpose(np.array(path))
    # plt.plot(cols, rows, '-')
    # plt.show()
    
while len(horstarts):
    path = deque()

    start = list(horstarts.keys())[0]
    end = horstarts.pop(start, None)
    horends.pop(end)
    path.append(start)
    path.append(end)
    
    add_connected_lines(connected_point=start, path=path, add_to_front=True, find_horizontal=False)
    add_connected_lines(connected_point=end, path=path, add_to_front=False, find_horizontal=False)

    paths.append(path)
    # print('path', path)

    # plt.xlim(0, 100)
    # plt.ylim(100, 0)
    # rows, cols = np.transpose(np.array(path))
    # plt.plot(cols, rows, '-')
    # plt.show()
    

print(len(verstarts))
print(len(horstarts))

# print(singles)
for s in singles:
    paths.append(deque([tuple(s)]))

# print(paths)


# al = np.array([(i[0], i[-1]) for i in paths])
# print(al.shape)

# gen[tuple(np.transpose(al[:, 0, :]))] = [255,0,0]
# gen[tuple(np.transpose(al[:, 1, :]))] = [0,255,0]



# step 1
# in img find 2 connected diagonal pattern
# something like: 
# 0 0 0
# 0 A 0
# 0 0 B
# only when there is only two connected points and nothing else to the center of A

# step 2
# once we find B
# if A is end and B is start add B to right of A
# if A is end and B is end add reverse of B to right of A
# if A is start and B is start add reverse of B to left of A
# if A is start and B is end add B to left of A

# step 3
# remove A from img
# remove B from img if length of path B is not 1

# step 4
# repeat from step 1 until no pattern match is found


starts = {path[0]: path for path in paths}
ends = {path[-1]: path for path in paths}

assert len(starts) == len(ends)

indexes = list(zip(starts.keys(), ends.keys()))


def connect_diagonal(to_start, connected_point):
    curr_start, curr_end = indexes[0][0], indexes[0][1]

    # Don't allow connecting a path to itself
    if np.array_equal(connected_point, curr_start) or np.array_equal(connected_point, curr_end):
        return False

    def _connect_path(find_start):
        global indexes
        found_path = (starts if find_start else ends).pop(connected_point, None)

        if found_path:
            # We need to reverse the array when we find a start connected to a start
            # or when we find an end connected to an end
            if (to_start and find_start) or (not to_start and not find_start):
                found_path.reverse()

            # We also need to remove the path from the twin dict
            (ends if find_start else starts).pop(found_path[0 if to_start else -1])
            if to_start:
                # Add the current path to the right of what we found
                found_path.extend(starts[curr_start])
                new_start = found_path[0]
                # The start point has changed so remove it
                starts.pop(curr_start)
                # The path in starts and ends need to mirror each other
                starts[new_start] = found_path
                ends[curr_end] = found_path
                # Overwrite the start value of the current path
                indexes[0] = (new_start, curr_end)
                # The path that we found should no longer be used so remove it from indexes
                found_index = (connected_point, new_start) if find_start else (new_start, connected_point)
                indexes = [i for i in indexes if i != found_index]
            else:
                # Add the path that we found to the right of the current path
                starts[curr_start].extend(found_path)
                new_end = found_path[-1]
                # The end point has changed so remove it
                ends.pop(curr_end)
                # The path in ends needs to mirror the path in starts
                ends[new_end] = starts[curr_start]
                # Overwrite the end value of the current path
                indexes[0] = (curr_start, new_end)
                # The path that we found should no longer be used so remove it from indexes
                found_index = (connected_point, new_end) if find_start else (new_end, connected_point)
                indexes = [i for i in indexes if i != found_index]

            return True

        return False

    # Try to find a path with a start connected to our point
    if _connect_path(find_start=True):
        return True

    # If we couldn't find a connected start point try to find a connected end point
    return _connect_path(find_start=False)



# def connect_diagonal_to_start(connected_point):
#     curr_start, curr_end = indexes[0][0], indexes[0][1]

#     # Don't allow connecting a path to itself
#     if np.array_equal(connected_point, curr_start) or np.array_equal(connected_point, curr_end):
#         return False

#     def _connect_path(find_start):
#         global indexes
#         found_path = (starts if find_start else ends).pop(connected_point, None)

#         if found_path:
#             if find_start:
#                 found_path.reverse()
#             # We also need to remove the path from the twin dict
#             (ends if find_start else starts).pop(found_path[0])
#             # Add the current path to the right of what we found
#             found_path.extend(starts[curr_start])
#             new_start = found_path[0]
#             # The start point has changed so remove it
#             starts.pop(curr_start)
#             # The path in starts and ends need to mirror each other
#             starts[new_start] = found_path
#             ends[curr_end] = found_path
#             # Overwrite the start value of the current path
#             indexes[0] = (new_start, curr_end)
#             # The path that we found should no longer be used so remove it from indexes
#             found_index = (connected_point, new_start) if find_start else (new_start, connected_point)
#             print('found_index', found_index)
#             indexes = [i for i in indexes if i != found_index]

#             return True

#         return False

#     # Try to find a path with a start connected to our point
#     if _connect_path(find_start=True):
#         return True

#     # If we couldn't find a connected start point try to find a connected end point
#     return _connect_path(find_start=False)


# def connect_diagonal_to_end(connected_point):
#     curr_start, curr_end = indexes[0][0], indexes[0][1]

#     # Don't allow connecting a path to itself
#     if np.array_equal(connected_point, curr_start) or np.array_equal(connected_point, curr_end):
#         return False

#     def _connect_path(find_start):
#         global indexes
#         found_path = (starts if find_start else ends).pop(connected_point, None)

#         if found_path:
#             if not find_start:
#                 found_path.reverse()
#             # We also need to remove the path from the twin dict
#             (ends if find_start else starts).pop(found_path[-1])
#             # Add the path that we found to the right of the current path
#             starts[curr_start].extend(found_path)
#             new_end = found_path[-1]
#             # The end point has changed so remove it
#             ends.pop(curr_end)
#             # The path in ends needs to mirror the path in starts
#             ends[new_end] = starts[curr_start]
#             # Overwrite the end value of the current path
#             # print(indexes)
#             indexes[0] = (curr_start, new_end)
#             # The path that we found should no longer be used so remove it from indexes
#             found_index = (connected_point, new_end) if find_start else (new_end, connected_point)
#             indexes = [i for i in indexes if i != found_index]

#             return True

#         return False

#     # Try to find a path with a start connected to our point
#     if _connect_path(find_start=True):
#         return True

#     # If we couldn't find a connected start point try to find a connected end point
#     return _connect_path(find_start=False)        

while len(indexes):
    found_start_extremity = False
    found_end_extremity = False

    while not found_start_extremity:
        start = indexes[0][0]

        # Top Left
        connected_point = (start[0]-1, start[1]-1)
        if connect_diagonal(to_start=True, connected_point=connected_point):
            continue

        # Top Right
        connected_point = (start[0]-1, start[1]+1)
        if connect_diagonal(to_start=True, connected_point=connected_point):
            continue

        # Bottom Left
        connected_point = (start[0]+1, start[1]-1)
        if connect_diagonal(to_start=True, connected_point=connected_point):
            continue

        # Bottom Right
        connected_point = (start[0]+1, start[1]+1)
        if connect_diagonal(to_start=True, connected_point=connected_point):
            continue

        # If we couldn't find any paths connected to this start point we have found the extremity
        # of this path so exit the loop and try to find a connected path to the end point
        found_start_extremity = True


    while not found_end_extremity:
        end = indexes[0][1]

        # Top Left
        connected_point = (end[0]-1, end[1]-1)
        if connect_diagonal(to_start=False, connected_point=connected_point):
            continue

        # Top Right
        connected_point = (end[0]-1, end[1]+1)
        if connect_diagonal(to_start=False, connected_point=connected_point):
            continue

        # Bottom Left
        connected_point = (end[0]+1, end[1]-1)
        if connect_diagonal(to_start=False, connected_point=connected_point):
            continue

        # Bottom Right
        connected_point = (end[0]+1, end[1]+1)
        if connect_diagonal(to_start=False, connected_point=connected_point):
            continue

        # If we couldn't find any paths connected to this end point we have found the extremity
        # of this path so exit the loop and move on to the next path on the list
        found_end_extremity = True

    # There are no more paths that can connect to this one so remove it and go to the next
    indexes = indexes[1:]


assert len(starts) == len(ends)
assert set([tuple(path) for path in starts.values()]) == set([tuple(path) for path in ends.values()])


paths = [tuple(path) for path in starts.values()]


# for i, path in enumerate(paths):
#     rows, cols = np.transpose(path)
#     plt.xlim(0, 100)
#     plt.ylim(100, 0)
#     plt.plot(cols, rows, '-')
# plt.show()



# for i in range(len(starts)):
    # print(i)
    # start = 

# while curr_path < len(paths):
#     print(curr_path)
#     found_start_extremity = False
#     found_end_extremity = False

#     while not found_start_extremity:
#         start = paths[curr_path][0]
        
#         found_start_extremity = True

#     while not found_end_extremity:
#         found_end_extremity = True


#     curr_path += 1


# print('len', len(paths))


# while ver lines still not empty:
#     ver_line = get first line from data

#     add ver_line to group
#     remove ver_line from vertical lines

    
#     nextype = 'horizontal'
#     set last_connected to startpoint

#     while True:
#         if nextype is 'vertical':
#             ver = find_connected(last_connected, vertical lines)
#             if ver:
#                 insert ver to start in group
#                 remove ver from verticals
#                 set last_connected to ?
#                 nextype = 'horizontal'
#             else:
#                 break

#         if nextype is 'horizontal':
#             hor = find_connected(last_connected, horizontal lines)
#             if hor:
#                 insert hor to start in group
#                 remove hor from horizontals
#                 set last_connected to ?
#                 nextype = 'vertical'
#             else:
#                 break

#     nextype = 'horizontal'
#     set last_connected to endpoint

#     while True:
#         if nextype is 'vertical':
#             ver = find_connected(last_connected, vertical lines)
#             if ver:
#                 insert ver to end in group
#                 remove ver from verticals
#                 set last_connected
#                 nextype = 'horizontal'
#             else:
#                 break

#         if nextype is 'horizontal':
#             hor = find_connected(last_connected, horizontal lines)
#             if hor:
#                 insert hor to end in group
#                 remove hor from horizontals
#                 set last_connected
#                 nextype = 'vertical'
#             else:
#                 break













def to_svg(paths, width, height):
    stroke_width = 1
    offset = stroke_width / 2

    paths_str = ''

    for path in paths:
        print()
            
        path = np.array(path, dtype=np.float32)
        
        # Reverse to go from y,x to x,y
        path = path[:, ::-1]

        # Handle single points
        if len(path) == 1:
            # Center x
            print('p',path)
            path[0, 0] += offset
            # Add an end anchor
            path = np.tile(path, [2, 1])
            # Offset y
            path[1, 1] += 1
            print('p',path)
        else:
            # Center the anchor
            path += offset

            # Place the start anchor of the path
            start_offset = path[0] - path[1]
            # Clip since we only wanna know if x/y is the same, smaller or bigger
            start_offset = np.clip(start_offset, -1, 1)
            start_offset *= offset
            path[0] += start_offset
            print('o', start_offset)

            # PLace the end anchor of the path
            end_offset = path[-1] - path[-2]
            end_offset = np.clip(end_offset, -1, 1)
            end_offset *= offset
            path[-1] += end_offset

            print(path)

            print('e', end_offset)

        anchors = ' L'.join([','.join(map(str, i)) for i in path[1:]])

        curr_path = f'\t<path stroke="#ff0000" fill="none" d="M{path[0, 0]},{path[0, 1]} L{anchors}" />\n'
        paths_str = ''.join([paths_str, curr_path])

    svg_begin = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
    svg_end = '</svg>\n'
    svg = ''.join([svg_begin, paths_str, svg_end])

    return svg

svg = to_svg(paths, 100, 100)
print('svg\n', svg)


with open('resultskate.svg', 'w') as f:
    f.write(svg)


# a = [13, 83]
# print(a in ver)
# while len(ver):
#     line = ver[0]
#     ver = ver[1:]


# paths = []

# for start, end in cps_ver:
#     gen[tuple(start)] = [1, 0, 0]
#     gen[tuple(end)] = [0, 1, 0]
#     path = f'<path stroke="rgb(255,0,0)" fill="none" d="M{start[1]+.5},{start[0]} L{end[1]+.5},{end[0]+1}" />\n'
#     paths.append(path)

# for start, end in cps_hor:
#     gen[tuple(start)] = [1, 0.7, 0]
#     gen[tuple(end)] = [0, 0.7, 0.7]
#     path = f'<path stroke="rgb(0,255,0)" fill="none" d="M{start[1]},{start[0]+.5} L{end[1]+1},{end[0]+.5}" />\n'
#     paths.append(path)

# for row, col in singles:
#     gen[row, col] = [1, 0.5, 0.5]
#     path = f'<path stroke="rgb(0,0,255)" fill="none" d="M{col+.5},{row} L{col+.5},{row+1}" />\n'
#     paths.append(path)

# width = 100
# height = 100

# svg_begin = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
# svg_end = '</svg>\n'
# svg = ''.join([svg_begin, *paths, svg_end])

# with open('resultskate.svg', 'w') as f:
    # f.write(svg)

# plt.figure(figsize=(10,10))
# plt.imshow(gen)
# plt.show()

end_time = time.time() - start_time
print('Done! - Took: ', end_time, 's')

# 0 0 0 1 1 1 0 0 0
#   0 0 1 0 0 -1 0 0

#%%
import cairosvg
import tensorflow as tf

img = cairosvg.svg2png(svg.encode('utf-8'))
img = tf.io.decode_jpeg(img, channels=3)

plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()

