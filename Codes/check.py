from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2

# Paths to the 'train' folder and 'parse' folder
train_image_folder = 'C:\\Project\\fashion\\data_split\\train'
parse_folder = os.path.join(train_image_folder, 'parse')  # Assuming parsed images are in the train folder
info_folder = 'C:\\Project\\fashion\\info'  # Path to the info folder
output_folder = 'C:\\Project\\fashion\\output'  # Path to the output folder

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

output_display_counter = 0

# Initialize a counter for saved output images
output_counter = 0

# Iterate through all image files in the train folder
for image_name in os.listdir(train_image_folder):
    if image_name.endswith(".jpg"):  # Check that the file is a .jpg image
        basename = image_name[:-4]
        

        # Create the parsed image name by replacing '.jpg' with '_segm.png'
        parsed_image_name = image_name.replace('.jpg', '_segm.png')
        parsed_image_path = os.path.join(parse_folder, parsed_image_name)

        # Check if the parsed image exists in the parse folder
        if os.path.exists(parsed_image_path):
            # Construct the paths to the corresponding text files
            keypoints_loc_path = os.path.join(info_folder, 'keypoints_loc.txt')
            keypoints_vis_path = os.path.join(info_folder, 'keypoints_vis.txt')

            # Read the keypoints and visibility
            keypoints_dict = {}
            with open(keypoints_loc_path, 'r') as keypoints_loc_file:
                for line in keypoints_loc_file:
                    parts = line.strip().split()
                    filename = parts[0]
                    keypoints = [int(x) for x in parts[1:]]
                    keypoints_dict[filename] = keypoints

            visibility_dict = {}
            with open(keypoints_vis_path, 'r') as keypoints_vis_file:
                for line in keypoints_vis_file:
                    parts = line.strip().split()
                    filename = parts[0]
                    visibility = [int(x) for x in parts[1:]]
                    visibility_dict[filename] = visibility

            # Get the keypoints and visibility for the current image
            keypoints_loc = keypoints_dict[basename + ".jpg"]
            keypoints_vis = visibility_dict[basename + ".jpg"]

            # Open the original image
            original_image = Image.open(os.path.join(train_image_folder, image_name))
            original_image_array = np.array(original_image)

            # Draw the keypoints
            for i in range(0, len(keypoints_loc), 2):
                x, y = keypoints_loc[i], keypoints_loc[i + 1]
                vis = keypoints_vis[i // 2]
                if vis == 0:  # If the keypoint is visible
                    cv2.circle(original_image_array, (x, y), 5, (0, 255, 0), -1)
                elif vis == 1:  # If the keypoint is present but hidden
                    cv2.circle(original_image_array, (x, y), 5, (0, 0, 255), -1)

                    

            # Convert the NumPy array back to a PIL Image
            original_image_with_keypoints = Image.fromarray(original_image_array)

            # Save the output image
            output_image_path = os.path.join(output_folder, f"{basename}_output.jpg")
            original_image_with_keypoints.save(output_image_path)
            print(f"Saved output image: {output_image_path}")

            # Increment the counter for saved output images
            output_counter += 1

            # Display up to 3 output images
            if output_counter <= 3:
                original_image_with_keypoints.show()

# Print the total number of saved output images
print(f"Total saved output images: {output_counter}")
