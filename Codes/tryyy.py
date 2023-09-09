from PIL import Image
import os

# Paths to the 'train' folder and 'parse' folder
train_image_folder = 'C:\\Project\\fashion\\data_split\\train'
parse_folder = os.path.join(train_image_folder, 'parse')  # Assuming parsed images are in the train folder
info_folder = 'C:\\Project\\fashion\\info'  # Path to the info folder

# Initialize a variable to keep track of the number of matching images
matching_count = 0

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
            shape_anno_all_path = os.path.join(info_folder, 'shape_anno_all.txt')
            fabric_ann_path = os.path.join(info_folder, 'fabric_ann.txt')
            pattern_ann_path = os.path.join(info_folder, 'pattern_ann.txt')

            # Check if all required text files have entries for the image
            with open(keypoints_loc_path, 'r') as keypoints_loc_file, \
                    open(keypoints_vis_path, 'r') as keypoints_vis_file, \
                    open(shape_anno_all_path, 'r') as shape_anno_all_file, \
                    open(fabric_ann_path, 'r') as fabric_ann_file, \
                    open(pattern_ann_path, 'r') as pattern_ann_file:
                if any(line.strip().startswith(basename) for line in keypoints_loc_file) and \
                   any(line.strip().startswith(basename) for line in keypoints_vis_file) and \
                   any(line.strip().startswith(basename) for line in shape_anno_all_file) and \
                   any(line.strip().startswith(basename) for line in fabric_ann_file) and \
                   any(line.strip().startswith(basename) for line in pattern_ann_file):
                    matching_count += 1
                    print(f"Image {basename} in train folder has a corresponding parsed image and matching text file entries.")
                else:
                    print(f"Image {basename} in train folder has a parsed image but not all required text file entries.")
        else:
            print(f"Image {basename} in train folder does not have a corresponding parsed image.")

# Print the total number of matching images
print(f"Total matching images: {matching_count}")
