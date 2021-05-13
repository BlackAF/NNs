#%%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import cairosvg
import tensorflow as tf
from collections import deque
import time

#%%
# ----- Canny -----

og_img = cv.imread('skate.jpg')
img = cv.cvtColor(og_img, cv.COLOR_BGR2GRAY)
# img = cv.GaussianBlur(img, (5, 5), 1)
canny = cv.Canny(img, 100, 200)

# cv.imwrite('skate_canny.jpg', canny)

plt.imshow(canny)
plt.show()

print('non zero', np.count_nonzero(canny))

canny_coords = np.transpose(canny.nonzero())#np.nonzero(canny)

print('canny_coords', canny_coords.shape)

# ret, thresh = cv.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

print(np.concatenate(contours).reshape(-1).shape)

print(len(contours), 'len')

contours_idx = None

c_img = np.zeros_like(og_img)
for i, contour in enumerate(contours[:contours_idx]):
    # print('-----')
#     # print(contour.shape)
    # c_img = np.zeros_like(og_img)
    c_img = cv.drawContours(c_img, contours, i, [255,255,255], 1)
    # c_img = cv.drawContours(c_img, contours, i, np.random.randint(0, 256, size=3).tolist(), 1)
    # plt.figure(figsize=(10,10))
    # plt.imshow(c_img)


# plt.figure(figsize=(10,10))
plt.imshow(c_img)
plt.show()
print('done')




# begin_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">\n'

# paths = []

# for contour in contours[:contours_idx]:
#     color = ','.join(map(str, np.random.randint(0, 256, size=3)))
#     path = f'\t<path fill="none" stroke="rgb({color})" stroke-width="1px" d="M'
#     for anchor in contour:
#         x, y = anchor[0]
#         path += f' {x},{y}'
#     path += '" />\n'
    
#     paths.append(path)

# end_svg = '</svg>\n'

# svg = ''.join([begin_svg, *paths, end_svg])

# with open('contours.svg', 'w') as f:
#     f.write(svg)

# # print(svg)


# svg_img = cairosvg.svg2png(svg)
# svg_img = tf.io.decode_jpeg(svg_img, channels=3)

# plt.imshow(svg_img)
# plt.show()
# print('done2')

# %%
# ----- Threshold -----

og_img = cv.imread('skate_highres.jpg')
img = cv.cvtColor(og_img, cv.COLOR_BGR2GRAY)
# img = cv.GaussianBlur(img, (5, 5), 1)
# img = cv.medianBlur(img, 5)

ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, th4 = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

plt.figure(figsize=(10,10))
# plt.imshow(img, cmap='gray')
# plt.imshow(th1, cmap='gray')
plt.imshow(th2, cmap='gray')
# plt.imshow(th3, cmap='gray')
# plt.imshow(th4, cmap='gray')
plt.show()

# %%
# ----- Sharpen -----

# get parent shapes: adaptivethreshold 21 2, denoising 110 7 21 x3, erode iterations 5, canny 0 40, dilate iterations 5

og_img = cv.imread('skate_highres.jpg')
img = cv.cvtColor(og_img, cv.COLOR_BGR2GRAY)
# img = cv.GaussianBlur(img, (5, 5), 1)
# img = cv.medianBlur(img, 5)

img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 2)
img = cv.fastNlMeansDenoising(img, None, 110, 7, 21)
img = cv.fastNlMeansDenoising(img, None, 110, 7, 21)
img = cv.fastNlMeansDenoising(img, None, 110, 7, 21)
# img = cv.fastNlMeansDenoising(img, None, 110, 7, 21)
# img = cv.fastNlMeansDenoising(img, None, 110, 7, 21)
# img = cv.fastNlMeansDenoising(img, None, 50, 3, 11)

img = cv.erode(img, np.ones((3, 3), np.uint8), iterations=5)
img = cv.Canny(img, 0, 40)
# kernel = np.ones((3, 3), np.uint8)
img = cv.dilate(img, np.ones((3, 3), np.uint8), iterations=5)
# img = cv.dilate(img, kernel, iterations=5)
# img = cv.erode(img, kernel, iterations=2)


# imgc = np.zeros(img.shape+(3,))
imgc = np.zeros(img.shape)
contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
hierarchy = hierarchy[0]
for i, cnt in enumerate(contours):
    # if hierarchy[i][2] != -1 and cv.contourArea(cnt) > cv.arcLength(cnt, True):
    # if hierarchy[i][2] != -1 and cv.contourArea(cnt) >= 0:
    # if hierarchy[i][2] != -1:
    # if hierarchy[i][2] == -1 and cv.arcLength(cnt, True) > 150:
    if hierarchy[i][3] == -1:
    # if hierarchy[i][3] == -1 or hierarchy[i][3] == 0:
        # print('i', i)
        cv.drawContours(imgc, contours, i, 255, 1)
        # cv.drawContours(imgc, contours, i, np.random.randint(0, 256, size=3).tolist(), 3)

# kernel = np.ones((3, 3), np.uint8)
# imgc = cv.dilate(imgc, kernel, iterations=2)
# imgc = cv.erode(imgc, kernel, iterations=1)

# mask = np.zeros(img.shape[:2],np.uint8)
# cv.drawContours(mask, [cnt],-1, 255, -1)
# dst = cv.bitwise_and(og_img, og_img, mask=mask)

plt.figure(figsize=(10,10))
plt.imshow(imgc, 'gray')
plt.show()

# cv.imwrite('skate_canny_2000.jpg', imgc)


#%%


rows, cols = np.nonzero(imgc)

# plt.figure(figsize=(10,10))
# plt.xlim(0, 2000)
# plt.ylim(2000, 0)
# plt.plot(cols[:100], rows[:100])
# plt.plot(cols, rows)
# plt.show()

#%%
og_img = cv.imread('skate_highres.jpg')
img = cv.cvtColor(og_img, cv.COLOR_BGR2HSV)
# img = cv.medianBlur(img, 5)
# img = cv.cvtColor(og_img, cv.COLOR_BGR2RGB)
# img = cv.GaussianBlur(img, (5, 5), 1)
img = cv.fastNlMeansDenoising(img, None, 60, 5, 13)
img = cv.fastNlMeansDenoising(img, None, 60, 5, 13)
img = cv.erode(img, np.ones((3, 3), np.uint8), iterations=2)
img = cv.dilate(img, np.ones((3, 3), np.uint8), iterations=2)
img = cv.Canny(img, 20, 50)

imgc = np.zeros(img.shape+(3,))
contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(imgc, contours, -1, [255,255,255], 3)


plt.figure(figsize=(10,10))
# plt.imshow(img)
plt.imshow(imgc, 'gray')
plt.show()

#%%
# ---- Remove Background ----
og_img = cv.imread('skate_highres.jpg')
img = cv.cvtColor(og_img, cv.COLOR_BGR2GRAY)

img = cv.erode(img, np.ones((3, 3), np.uint8), iterations=1)
img = cv.Canny(img, 20, 80)
img = cv.dilate(img, np.ones((3, 3), np.uint8), iterations=15)

contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
hierarchy = hierarchy[0]
mask = np.zeros_like(img)
for i, cnt in enumerate(contours):
    if hierarchy[i][3] == -1:
        cv.drawContours(mask, contours, i, [255,255,255], -1)

mask = cv.erode(mask, np.ones((3, 3), np.uint8), iterations=12)
og_img = cv.cvtColor(og_img, cv.COLOR_BGR2GRAY)
img = np.where(mask == 0, 255, og_img)

plt.figure(figsize=(10,10))
plt.imshow(img, 'gray')
# plt.imshow(img)
plt.show()

#%%
# ---- Lines ----
og_img = cv.imread('skate_highres.jpg')
img = cv.cvtColor(og_img, cv.COLOR_BGR2GRAY)

img = cv.erode(img, np.ones((3, 3), np.uint8), iterations=1)
img = cv.Canny(img, 20, 80)
img = cv.dilate(img, np.ones((3, 3), np.uint8), iterations=15)

contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
hierarchy = hierarchy[0]
mask = np.zeros_like(og_img)
for i, cnt in enumerate(contours):
    if hierarchy[i][3] == -1:
        cv.drawContours(mask, contours, i, [255,255,255], -1)

mask = cv.erode(mask, np.ones((3, 3), np.uint8), iterations=10)
img = np.where(mask == 0, 255, og_img)
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# img = cv.fastNlMeansDenoising(img, None, 10, 5, 13)
# img = cv.fastNlMeansDenoising(img, None, 50, 7, 13)
# img = cv.GaussianBlur(img, (5, 5), 3)
# ret, img = cv.threshold(img, 90, 255, cv.THRESH_BINARY)
# img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 2)
# img = cv.fastNlMeansDenoising(img, None, 80, 5, 13)
# img = cv.fastNlMeansDenoising(img, None, 80, 5, 13)
# img = cv.fastNlMeansDenoising(img, None, 80, 5, 13)

img = cv.fastNlMeansDenoising(img, None, 60, 5, 13)
img = cv.fastNlMeansDenoising(img, None, 60, 5, 13)
# img = cv.fastNlMeansDenoising(img, None, 30, 5, 13)
# img = cv.fastNlMeansDenoising(img, None, 110, 7, 21)
# img = cv.Canny(img, 20, 50)
img = cv.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
img = cv.erode(img, np.ones((3, 3), np.uint8), iterations=1)
# img = cv.GaussianBlur(img, (5, 5), 1)
img = cv.Canny(img, 20, 50)

# img = cv.bitwise_or(img1, img2)

# img = cv.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
# img = cv.fastNlMeansDenoising(img, None, 110, 5, 13)
# img = cv.Canny(img, 10, 50)

mask = mask[:, :, 0]
contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
maskline = np.zeros_like(img)
cv.drawContours(maskline, contours, -1, 255, 2)

img = cv.bitwise_or(img, maskline)

contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
img = np.zeros_like(img)
cv.drawContours(img, contours, -1, 255, 2)

# OpenCV adds a dumbass third dimension so get rid of it
# (can't use squeeze as it would also remove the first dimension in cases where there's only one point)
contours = [np.flip(contour[:, 0, :], 1) for contour in contours]


# plt.figure(figsize=(10,10))
# plt.xlim(0, 2000)
# plt.ylim(2000, 0)

# for c in contours:
#     c = np.transpose(c)
#     plt.plot(c[0], c[1], linewidth=1)

# plt.show()

# plt.figure(figsize=(10,10))
# plt.xlim(0, 2000)
# plt.ylim(2000, 0)
# # plt.imshow(img, 'gray')
# # plt.imshow(maskline, 'gray')
# plt.plot(np.transpose(contours[0])[1], np.transpose(contours[0])[0])
# # plt.hist(img.ravel(),256,[0,256])
# plt.show()

# cv.imwrite('skate_canny_2000.jpg', img)
# with open('contours.npy', 'wb') as f:
    # np.save(f, np.array(contours, dtype=object))


# %%
# ----- Canny -----
from sklearn.cluster import KMeans

og_img = cv.imread('skate_highres.jpg')
img = cv.cvtColor(og_img, cv.COLOR_BGR2RGB)
# img = cv.GaussianBlur(img, (5, 5), 1)
# canny = cv.Canny(img, 80, 50)

img = img.reshape(-1, 3)
img = img / 255

kmeans = KMeans(n_clusters=2, random_state=0).fit(img)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

img = pic2show.reshape(og_img.shape)

plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()

# %%
# ----- Canny -----
# og_img = cv.imread('skate.jpg')
og_img = cv.imread('skate_highres.jpg')
img = cv.cvtColor(og_img, cv.COLOR_BGR2GRAY)
# th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 31, 2)
# th2 = cv.fastNlMeansDenoising(th2, None, 110, 7, 21)
# th2 = cv.GaussianBlur(th2, (15, 15), 4)

# canny = cv.Canny(th2, 150, 500)

# ret,th1 = cv.threshold(th2, 70, 5, cv.THRESH_BINARY)


# contours, hierarchy = cv.findContours(canny, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
# hierarchy = hierarchy[0]


# closed_contours = []

# for i, cnt in enumerate(contours):
    # if hierarchy[i][2] != -1 and cv.contourArea(cnt) > cv.arcLength(cnt, True):
    # if hierarchy[i][2] == -1:
        # closed_contours.append(cnt)

# print('closed_contours', len(closed_contours))

# max_area = -1
# for i in range(len(contours)):
    # area = cv.contourArea(contours[i])
    # if area>max_area:
        # cnt = contours[i]
        # max_area = area


# imgc = np.zeros_like(img)
# cv.drawContours(imgc, closed_contours, -1, (255,255,255), 3)
# cv.drawContours(imgc, contours, -1, (255,255,255), 3)
# cv.drawContours(imgc, [cnt], 0, (255,255,255), 3)

# approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
# cv.drawContours(imgc, [approx], 0, (255,255,255), 3)


plt.figure(figsize=(10,10))
# plt.imshow(canny, 'gray')
# plt.imshow(imgc, 'gray')
# plt.imshow(th1, 'gray')
# plt.imshow(th2, 'gray')
plt.imshow(img, 'gray')
plt.show()

#%%
# --- Cluster ---
# og_img = cv.imread('skate.jpg')
# og_img = cv.imread('skate_highres.jpg')
# img = cv.cvtColor(og_img, cv.COLOR_BGR2GRAY)

# seed = (0, 0)
# seed = (20, 10)
seed = (230, 420)
# seed = (600, 1500)

# cv.floodFill(img, None, seedPoint=seed, newVal=(255, 0, 0), loDiff=40, upDiff=300)
# cv.floodFill(img, None, seedPoint=seed, newVal=(0, 0, 255), loDiff=(9, 5, 5, 5), upDiff=(5, 5, 5, 5))
# cv.circle(img, seed, 2, (0, 255, 255), cv.FILLED, cv.LINE_AA)
img = flood_fill(th2, seed, new_color=(255, 0, 0), threshold=38)


plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()

#%%
# ---- FLOOD FILL INITIAL ----

og_img = cv.imread('skate.jpg')
# og_img = cv.imread('skate_highres.jpg')
img = cv.cvtColor(og_img, cv.COLOR_BGR2GRAY)
num_rows, num_cols = img.shape

start = time.time()
img_dict = { (row, col): img[row, col] for col in range(img.shape[1]) for row in range(img.shape[0]) }
print('took - ', time.time() - start)

# seed = (0, 0)
# seed = (250, 420)
seed = (10, 20)
seed_value = img[seed]

print('SEED_VALUE', seed_value)

start = time.time()

to_visit = deque([seed])

flooded = set()
# visited = set()

def should_flood(point):
    threshold = 40
    point_value = img_dict.pop(point, None)
    return point_value is not None and seed_value - threshold <= point_value <= seed_value + threshold


# def should_flood(point):
    # threshold = 15
    # return seed_value - threshold <= img[point] <= seed_value + threshold

while len(to_visit) != 0:
    row, col = to_visit.popleft()
    # visited.add((row, col))
    # print(len(to_visit))

    if should_flood((row, col)):
        flooded.add((row, col))
        
        left = (row, col - 1)
        # if left not in visited and left[1] >= 0:
            # to_visit.append(left)
        
        top = (row - 1, col)
        # if top not in visited and top[0] >= 0:
            # to_visit.append(top)
        
        right = (row, col + 1)
        # if right not in visited and right[1] < num_cols:
            # to_visit.append(right)
        
        bottom = (row + 1, col)
        # if bottom not in visited and bottom[0] < num_rows:
            # to_visit.append(bottom)

        to_visit.extend([left, top, right, bottom])


# print(to_visit)
# print(flooded)

print('took - ', time.time() - start)

flooded = np.array([list(i) for i in flooded])
flooded = np.transpose(flooded)

img = np.expand_dims(img, axis=-1)
img = np.tile(img, [1, 1, 4])
img[:, :, -1] = 255

plt.figure(figsize=(10,10))

plt.imshow(img)

img[seed] = (255, 0, 0, 255)
img[flooded[0], flooded[1]] = (0, 255, 0, 150)

plt.imshow(img)
plt.show()


#%%
# ---- FLOOD FILL ----

def flood_fill(img, seed, new_color, threshold):
    flooded_img = img
    flooded_img = np.expand_dims(flooded_img, axis=-1)
    flooded_img = np.tile(flooded_img, [1, 1, 3])

    num_rows, num_cols = img.shape

    img_dict = { (row, col): img[row, col] for col in range(num_cols) for row in range(num_rows) }

    seed_value = img_dict[seed]

    to_visit = deque([seed])

    def should_flood(point):
        point_value = img_dict.pop(point, None)
        return point_value is not None and seed_value - threshold <= point_value <= seed_value + threshold

    while len(to_visit) != 0:
        row, col = to_visit.popleft()

        if should_flood((row, col)):
            flooded_img[(row, col)] = new_color
        
            left = (row, col - 1)
            top = (row - 1, col)
            right = (row, col + 1)
            bottom = (row + 1, col)

            to_visit.extend([left, top, right, bottom])

    return flooded_img