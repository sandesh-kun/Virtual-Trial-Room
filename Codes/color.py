from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2

# Paths to the text files containing the information
info_files = {
    'keypoints_loc': 'C:\\Project\\fashion\\info\\keypoints_loc.txt',
    'keypoints_vis': 'C:\\Project\\fashion\\info\\keypoints_vis.txt',
    'shape_anno_all': 'C:\\Project\\fashion\\info\\shape_anno_all.txt',
    'fabric_ann': 'C:\\Project\\fashion\\info\\fabric_ann.txt',
    'pattern_ann': 'C:\\Project\\fashion\\info\\pattern_ann.txt',
}

# Path to the 'new_image' and 'parse' folders
new_image_folder = 'C:\\Project\\fashion\\data_split\\train'
parse_folder = 'C:\\Project\\fashion\\data_split\\train\\parse'

# Define the image name
image_name = "MEN-Denim-id_00000353-02_7_additional.jpg"  # Update this with the image you want to process

# Read the information files
info_data = {}
for key, file_path in info_files.items():
    with open(file_path, 'r') as file:
        info_data[key] = {line.split()[0]: line.strip().split()[1:] for line in file}

# Open the original image
original_image = Image.open(os.path.join(new_image_folder, image_name))

# Create the parsed image name by replacing '.jpg' with '_segm.png' and open the parsed image
parsed_image_name = image_name.replace('.jpg', '_segm.png')
parsed_image = Image.open(os.path.join(parse_folder, parsed_image_name))
parsed_array = np.array(parsed_image)

# Create a blank image for annotations
output_image = Image.new("RGB", (original_image.width * 2, original_image.height))
output_image.paste(original_image, (0, 0))
output_image.paste(parsed_image, (original_image.width, 0))
draw = ImageDraw.Draw(output_image)
font = ImageFont.load_default()

# Segmentation Labels
segm_labels = [
    'background', 'top', 'outer', 'skirt', 'dress', 'pants', 'leggings', 'headwear', 'eyeglass', 'neckwear', 'belt',
    'footwear', 'bag', 'hair', 'face', 'skin', 'ring', 'wrist wearing', 'socks', 'gloves', 'necklace', 'rompers',
    'earrings', 'tie'
]

y_position = 10
for label_idx, label in enumerate(segm_labels):
    presence = (parsed_array == label_idx).any()
    draw.text((original_image.width + 10, y_position), f"{label}: {'Yes' if presence else 'No'}", font=font)
    y_position += 20

# Shape Annotations
shape_labels = [
    'sleeve length', 'lower clothing length', 'socks', 'hat', 'glasses',
    'neckwear', 'wrist wearing', 'ring', 'waist accessories', 'neckline',
    'outer clothing a cardigan?', 'upper clothing covering navel'
]
shape_values = info_data['shape_anno_all'].get(image_name[:-4], ["NA"] * 12)
for label_idx, label in enumerate(shape_labels):
    draw.text((original_image.width + 10, y_position), f"{label}: {shape_values[label_idx]}", font=font)
    y_position += 20

# Fabric Annotations
fabric_labels = ['upper fabric', 'lower fabric', 'outer fabric']
fabric_values = info_data['fabric_ann'].get(image_name[:-4], ["NA"] * 3)
for label_idx, label in enumerate(fabric_labels):
    draw.text((original_image.width + 10, y_position), f"{label}: {fabric_values[label_idx]}", font=font)
    y_position += 20

# Color Annotations
color_labels = ['upper color', 'lower color', 'outer color']
color_values = info_data['pattern_ann'].get(image_name[:-4], ["NA"] * 3)
for label_idx, label in enumerate(color_labels):
    draw.text((original_image.width + 10, y_position), f"{label}: {color_values[label_idx]}", font=font)
    y_position += 20

# Read the keypoints
keypoints_file = r'C:\Project\fashion\info\keypoints_loc.txt'
visibility_file = r'C:\Project\fashion\info\keypoints_vis.txt'

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

# Get the keypoints and visibility for the chosen image
keypoints_loc = keypoints_dict[os.path.basename(image_name)]
keypoints_vis = visibility_dict[os.path.basename(image_name)]

# Continue with your original code
original_image = Image.open(os.path.join(new_image_folder, image_name))
original_image_array = np.array(original_image)

# Draw the keypoints
for i in range(0, len(keypoints_loc), 2):
    x, y = keypoints_loc[i], keypoints_loc[i + 1]
    vis = keypoints_vis[i // 2]
    if vis == 0:  # If the keypoint is visible
        print(f'Drawing visible keypoint at ({x}, {y})')
        cv2.circle(original_image_array, (x, y), 5, (0, 255, 0), -1)
    elif vis == 1:  # If the keypoint is present but hidden
        print(f'Drawing hidden keypoint at ({x}, {y})')
        cv2.circle(original_image_array, (x, y), 5, (0, 0, 255), -1)
    else: # If the keypoint is not present
        print(f'Keypoint not present at ({x}, {y})')

# Convert the NumPy array back to a PIL Image
original_image_with_keypoints = Image.fromarray(original_image_array)

# Paste the image with keypoints into the final output image
output_image.paste(original_image_with_keypoints, (0, 0))

# Show or save the output image
output_image.show()
# output_image.save("output.png")
