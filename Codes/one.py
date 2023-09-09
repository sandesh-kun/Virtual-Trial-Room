import numpy as np
import os
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

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

# Number of images to process and save
num_images_to_process = 5

# Initialize empty lists to store X and Y data
X_data = []
Y_data = []

# List of image names in the train folder
image_names = [filename for filename in os.listdir(new_image_folder) if filename.endswith(".jpg")]

# Initialize a counter for processed images
images_processed = 0

info_data = {}
for key, file_path in info_files.items():
    with open(file_path, 'r') as file:
        info_data[key] = {line.split()[0]: line.strip().split()[1:] for line in file}

# Iterate through images
for image_name in image_names:
    if images_processed >= num_images_to_process:
        break

        # Load the original image
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
    keypoints_file = os.path.join('C:\\Project\\fashion\\info', 'keypoints_loc.txt')
    visibility_file = os.path.join('C:\\Project\\fashion\\info', 'keypoints_vis.txt')

    keypoints_dict = {}
    with open(keypoints_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            filename = parts[0]
            keypoints = [int(x) for x in parts[1:]]
            keypoints_dict[filename] = keypoints

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
    # Append the data to X and Y lists
    X_data.append(keypoints_loc)  # Append only keypoints_loc for X data
    Y_data.append(keypoints_vis)

    images_processed += 1

# Convert X and Y lists to NumPy arrays
X_data = np.array(X_data)
Y_data = np.array(Y_data)

# Create a DataFrame for X and Y data
num_keypoints = len(keypoints_loc) // 2  # Calculate the number of keypoints
X_columns = [f"keypoint_{i}_x" for i in range(num_keypoints)] + [f"keypoint_{i}_y" for i in range(num_keypoints)]
Y_columns = [f"visibility_{i}" for i in range(num_keypoints)]
X_df = pd.DataFrame(X_data, columns=X_columns)
Y_df = pd.DataFrame(Y_data, columns=Y_columns)

# Concatenate X and Y DataFrames
result_df = pd.concat([X_df, Y_df], axis=1)

# Save the data to a CSV file
csv_filename = "output_data.csv"
result_df.to_csv(csv_filename, index=False)

print(f"Saved data to {csv_filename}")
