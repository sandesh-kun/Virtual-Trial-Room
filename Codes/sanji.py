import cv2
import os

keypoints_file = r'C:\Project\fashion\info\keypoints_loc.txt'
visibility_file = r'C:\Project\fashion\info\keypoints_vis.txt'

# Read the keypoints
keypoints_dict = {}
with open(keypoints_file, 'r') as file:
    for line in file:
        parts = line.strip().split()
        filename = parts[0]
        keypoints = [int(x) for x in parts[1:]]
        keypoints_dict[filename] = keypoints

# Read the visibility
visibility_dict = {}
with open(visibility_file, 'r') as file:
    for line in file:
        parts = line.strip().split()
        filename = parts[0]
        visibility = [int(x) for x in parts[1:]]
        visibility_dict[filename] = visibility

# Choose an image to test
image_path = r'C:\Project\fashion\new_image\MEN-Sweaters-id_00000078-04_4_full.jpg'
image = cv2.imread(image_path)

# Get the keypoints and visibility for the chosen image
keypoints = keypoints_dict[os.path.basename(image_path)]
visibility = visibility_dict[os.path.basename(image_path)]

# Draw the keypoints
for i in range(0, len(keypoints), 2):
    x, y = keypoints[i], keypoints[i + 1]
    vis = visibility[i // 2]
    if vis == 0: # If the keypoint is visible
        print(f'Drawing keypoint at ({x}, {y})')
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

cv2.imshow('Keypoints', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
