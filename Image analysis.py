import cv2
import numpy as np
from google.colab.patches import cv2_imshow
# -----------------------------
# 1. Read image
# -----------------------------
img = cv2.imread("image.jpeg")   # replace with your image file
if img is None:
    raise Exception("Image not found!")
# -----------------------------
# 2. Show the image
# -----------------------------
cv2_imshow(img)
# -----------------------------
# 3. Change color to Black & White
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)
# -----------------------------
# 4. Image properties
# -----------------------------
height, width, channels = img.shape
file_size = img.nbytes / (1024 * 1024)

print("---- IMAGE PROPERTIES ----")
print("Width:", width)
print("Height:", height)
print("Channels:", channels)
print("Size (MB):", round(file_size, 2))
# -----------------------------
# 5. Rotate image (90, 180, 270)
# -----------------------------
rot_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rot_180 = cv2.rotate(img, cv2.ROTATE_180)
rot_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2_imshow(rot_90)
cv2_imshow(rot_180)
cv2_imshow(rot_270)
# -----------------------------
# 6. Mirror image (flip)
# -----------------------------
mirror_horizontal = cv2.flip(img, 1)  # left-right flip
mirror_vertical = cv2.flip(img, 0)    # up-down flip

cv2_imshow(mirror_horizontal)
cv2_imshow(mirror_vertical)
# -----------------------------
# 7. Find objects without deep learning
#    Using Canny Edge + Contours
# -----------------------------
edges = cv2.Canny(gray, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_objects = img.copy()
cv2.drawContours(img_objects, contours, -1, (0, 255, 0), 2)

cv2_imshow(img_objects)

print("Number of objects detected:", len(contours))
# -----------------------------
# 8. Cut image vertical 50-50
# -----------------------------
mid_x = width // 2
left_50 = img[:, :mid_x]
right_50 = img[:, mid_x:]

cv2_imshow(left_50)
cv2_imshow(right_50)
# -----------------------------
# 9. Cut image horizontal 50-50
# -----------------------------
mid_y = height // 2
top_50 = img[:mid_y, :]
bottom_50 = img[mid_y:, :]

cv2_imshow(top_50)
cv2_imshow(bottom_50)
# -----------------------------
# 10. Horizontal 70-30
# -----------------------------
cut_70 = int(height * 0.7)
top_70 = img[:cut_70, :]
bottom_30 = img[cut_70:, :]

cv2_imshow(top_70)
cv2_imshow(bottom_30)
# -----------------------------
# 11. Vertical 70-30
# -----------------------------
cut_70_v = int(width * 0.7)
left_70 = img[:, :cut_70_v]
right_30 = img[:, cut_70_v:]

cv2_imshow(left_70)
cv2_imshow(right_30)
# -----------------------------
# 12. Make small grid (4x4 or any size)
# -----------------------------
grid = img.copy()
h_step = height // 4
w_step = width // 4

for i in range(0, height, h_step):
    cv2.line(grid, (0, i), (width, i), (0, 255, 0), 1)

for j in range(0, width, w_step):
    cv2.line(grid, (j, 0), (j, height), (0, 255, 0), 1)

cv2_imshow(grid)
