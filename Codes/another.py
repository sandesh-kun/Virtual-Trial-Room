from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2
import pandas as pd

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

# List of image names in the train folder
image_names = [filename for filename in os.listdir(new_image_folder) if filename.endswith(".jpg")]

# Initialize an empty list to store image information
data = []

# Read the information files
info_data = {}
for key, file_path in info_files.items():
    with open(file_path, 'r') as file:
        info_data[key] = {line.split()[0]: line.strip().split()[1:] for line in file}

# Iterate through images
for image_name in image_names:
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

    # ... (Segmentation Labels, Shape Annotations, Fabric Annotations, Color Annotations)

    # Read the keypoints and visibility information
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

    # Continue with drawing keypoints
    original_image_array = np.array(original_image)
    # ... (Drawing keypoints and visibility)

    # Append the image information to the data list
    image_info = {
        'ImageName': image_name,
        'KeyPointsLoc': keypoints_loc,
        'KeyPointsVis': keypoints_vis,
        # ... (Other annotations, fabric, color, etc.)
    }
    data.append(image_info)

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_filename = "image_info.csv"
df.to_csv(csv_filename, index=False)

print(f"Saved data to {csv_filename}")
