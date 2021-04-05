#%%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import cairosvg
import tensorflow as tf

#%%

og_img = cv.imread('skate.jpg')
img = cv.cvtColor(og_img, cv.COLOR_BGR2GRAY)
# img = cv.GaussianBlur(img, (5, 5), 1)
canny = cv.Canny(img, 100, 200)

plt.imshow(canny)
plt.show()

print('non zero', np.count_nonzero(canny))

canny_coords = np.transpose(canny.nonzero())#np.nonzero(canny)

print('canny_coords', canny_coords.shape)

# ret, thresh = cv.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

print(np.concatenate(contours).reshape(-1).shape)

contours_idx = -1

c_img = np.zeros_like(og_img)
for i, contour in enumerate(contours[:contours_idx]):
    # print('-----')
#     # print(contour.shape)
    c_img = cv.drawContours(c_img, contours, i, np.random.randint(0, 256, size=3).tolist(), 1)

plt.imshow(c_img)
plt.show()
print('done')




begin_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">\n'

paths = []

for contour in contours[:contours_idx]:
    color = ','.join(map(str, np.random.randint(0, 256, size=3)))
    path = f'\t<path fill="none" stroke="rgb({color})" stroke-width="1px" d="M'
    for anchor in contour:
        x, y = anchor[0]
        path += f' {x},{y}'
    path += '" />\n'
    
    paths.append(path)

end_svg = '</svg>\n'

svg = ''.join([begin_svg, *paths, end_svg])

with open('contours.svg', 'w') as f:
    f.write(svg)

# print(svg)


svg_img = cairosvg.svg2png(svg)
svg_img = tf.io.decode_jpeg(svg_img, channels=3)

plt.imshow(svg_img)
plt.show()
print('done2')

# %%

# %%
