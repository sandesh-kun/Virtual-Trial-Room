import os
import shutil

# Path to the 'parse' folder and the images folder
parse_folder = 'C:\\Project\\fashion\\parse'
images_folder = 'C:\\Project\\fashion\\images'
output_folder = 'C:\\Project\\fashion\\new_image'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the list of files in the 'parse' folder
parse_files = os.listdir(parse_folder)

# Copy corresponding images from the 'images' folder to the output folder
for file_name in parse_files:
    # Get the image file name by replacing '_segm.png' with '.jpg'
    image_name = file_name.replace('_segm.png', '.jpg')

    # Check if the corresponding image exists in the 'images' folder
    image_path = os.path.join(images_folder, image_name)
    if os.path.exists(image_path):
        # Copy the image to the output folder
        output_image_path = os.path.join(output_folder, image_name)
        shutil.copy(image_path, output_image_path)
        print(f"Image '{image_name}' copied to '{output_folder}'")
    else:
        print(f"Corresponding image not found for '{file_name}'")
