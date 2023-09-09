import os
import shutil

# Source directories
images_directory = 'C:\\Project\\fashion\\images'
densepose_directory = 'C:\\Project\\fashion\\densepose'
segm_directory = 'C:\\Project\\fashion\\segm'

# Destination directories
destination_images_directory = 'C:\\Project\\fashion\\new_image'
destination_segm_directory = 'C:\\Project\\fashion\\parse'
destination_densepose_directory = 'C:\\Project\\fashion\\pose'

# Create destination directories if they don't exist
if not os.path.exists(destination_images_directory):
    os.makedirs(destination_images_directory)
if not os.path.exists(destination_segm_directory):
    os.makedirs(destination_segm_directory)
if not os.path.exists(destination_densepose_directory):
    os.makedirs(destination_densepose_directory)

# Counter for matching files
matching_count = 0

# Iterate through the files in the images directory
for filename in os.listdir(images_directory):
    if filename.endswith('.jpg'):
        # Construct the corresponding densepose and segm filenames
        densepose_filename = filename[:-4] + '_densepose.png'
        segm_filename = filename[:-4] + '_segm.png'

        # Construct full file paths
        image_path = os.path.join(images_directory, filename)
        densepose_path = os.path.join(densepose_directory, densepose_filename)
        segm_path = os.path.join(segm_directory, segm_filename)

        # Check if all three images exist
        if os.path.exists(densepose_path) and os.path.exists(segm_path):
            # Copy images to the corresponding destination folders
            shutil.copy(image_path, destination_images_directory)
            shutil.copy(densepose_path, destination_densepose_directory)
            shutil.copy(segm_path, destination_segm_directory)
            # Increment the counter
            matching_count += 1

# Print the result
print(f"Number of matching files copied: {matching_count}")
