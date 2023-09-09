from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2

# Paths to the text files containing the information
info_folder = 'C:\\Project\\fashion\\info'
data_split_folder = 'C:\\Project\\fashion\\data_split'
output_folder = 'C:\\Project\\fashion\\output'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# Read the other information files (not keypoints)
info_data = {}
for key in ['shape_anno_all', 'fabric_ann', 'pattern_ann']:
    with open(os.path.join(info_folder, f'{key}.txt'), 'r') as file:
        info_data[key] = {line.split()[0]: line.strip().split()[1:] for line in file}

# Iterate over images in the train folder
train_image_folder = os.path.join(data_split_folder, 'train')
train_parse_folder = os.path.join(data_split_folder, 'train', 'parse')
train_images = os.listdir(train_image_folder)

for image_name in train_images:
    if image_name.endswith('.jpg'):
        basename = image_name[:-4]
        image_path = os.path.join(train_image_folder, image_name)
        parsed_image_name = image_name.replace('.jpg', '_segm.png')
        parsed_image_path = os.path.join(train_parse_folder, parsed_image_name)

        # Open the original image and parsed image
        original_image = Image.open(image_path)
        parsed_image = Image.open(parsed_image_path)
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
        shape_labels = ['sleeve length', 'lower clothing length', 'socks', 'hat', 'glasses',
                        'neckwear', 'wrist wearing', 'ring', 'waist accessories', 'neckline',
                        'outer clothing a cardigan?', 'upper clothing covering navel']
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

        # Read keypoints and visibility
        keypoints_file = os.path.join(data_split_folder, 'train', 'keypoints_loc.txt')
        visibility_file = os.path.join(data_split_folder, 'train', 'keypoints_vis.txt')

        # Now, move the file-reading code for keypoints inside the loop
        keypoints_dict = {}
        with open(os.path.join(info_folder, 'keypoints_loc.txt'), 'r') as keypoints_loc_file:
            for line in keypoints_loc_file:
                parts = line.strip().split()
                filename = parts[0]
                keypoints = [int(x) for x in parts[1:]]
                keypoints_dict[filename] = keypoints

        visibility_dict = {}
        with open(os.path.join(info_folder, 'keypoints_vis.txt'), 'r') as file:
            for line in file:
                parts = line.strip().split()
                filename = parts[0]
                visibility = [int(x) for x in parts[1:]]
                visibility_dict[filename] = visibility

        # Continue with your original code for drawing keypoints
        keypoints_loc = keypoints_dict.get(basename + ".jpg", [])
        keypoints_vis = visibility_dict.get(basename + ".jpg", [])

        # Continue with your original code for drawing keypoints
        original_image_array = np.array(original_image)
        for i in range(0, len(keypoints_loc), 2):
            x, y = keypoints_loc[i], keypoints_loc[i + 1]
            vis = keypoints_vis[i // 2]
            if vis == 0:  # If the keypoint is visible
                cv2.circle(original_image_array, (x, y), 5, (0, 255, 0), -1)
            elif vis == 1:  # If the keypoint is present but hidden
                cv2.circle(original_image_array, (x, y), 5, (0, 0, 255), -1)
            else:  # If the keypoint is not present
                pass

        # Convert the NumPy array back to a PIL Image
        original_image_with_keypoints = Image.fromarray(original_image_array)

        # Paste the image with keypoints into the final output image
        output_image.paste(original_image_with_keypoints, (0, 0))

        # Show or save the output image
        output_image.show()
        output_image.save(os.path.join(output_folder, f'{image_name}_output.png'))
