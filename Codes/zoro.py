import os

# Path to the 'new_image' folder
new_image_folder = 'C:\\Project\\fashion\\new_image'

# List of paths to the .txt files you want to filter
txt_files = [
    'C:\\Project\\fashion\\keypoints\\keypoints_loc.txt',
    'C:\\Project\\fashion\\keypoints\\keypoints_vis.txt',
    'C:\\Project\\fashion\\labels\\shape\\shape_anno_all.txt',
    'C:\\Project\\fashion\\labels\\texture\\fabric_ann.txt',
    'C:\\Project\\fashion\\labels\\texture\\pattern_ann.txt'
]

# Destination directory for the new .txt files
destination_directory = 'C:\\Project\\fashion\\info'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Get a set of filenames from the 'new_image' folder
image_filenames = set(os.listdir(new_image_folder))

# Create a set to hold the unique matching names
unique_matches = set()

# Iterate through the .txt files
for txt_file_path in txt_files:
    # Get the filename of the current .txt file
    txt_filename = os.path.basename(txt_file_path)

    # Open the .txt file for reading
    with open(txt_file_path, 'r') as file:
        # Read lines and filter those whose filenames match the images in the 'new_image' folder
        filtered_lines = [line for line in file if line.split()[0] in image_filenames]
        # Add the matching names to the set of unique matches
        unique_matches.update([line.split()[0] for line in filtered_lines])

    # Create the path for the new .txt file in the destination directory
    new_txt_path = os.path.join(destination_directory, txt_filename)

    # Write the filtered lines to the new .txt file
    with open(new_txt_path, 'w') as file:
        file.writelines(filtered_lines)

    print(f"Saved filtered content to {new_txt_path}")

# Print the total number of unique matching names
print(f"Total unique matching names found: {len(unique_matches)}")
