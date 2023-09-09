import os
import shutil

# List of text files in the info folder
text_files = ['keypoints_loc.txt', 'keypoints_vis.txt', 'shape_anno_all.txt', 'fabric_ann.txt', 'pattern_ann.txt']

# Source and destination folders
info_folder = 'C:/Project/fashion/info'
data_split_folder = 'C:/Project/fashion/data_split'
test_folder = os.path.join(data_split_folder, 'test')
train_folder = os.path.join(data_split_folder, 'train')
val_folder = os.path.join(data_split_folder, 'val')

# Function to split the content of the text files and save to new text files
def split_and_save_text_files(file_path, save_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content by lines
    lines = content.split('\n')

    # Create new text file in the save_path directory
    new_file_path = os.path.join(save_path, os.path.basename(file_path))
    with open(new_file_path, 'w') as new_file:
        for line in lines:
            if line:
                file_name, data = line.split(' ', 1)
                file_name = file_name.strip()
                new_file.write(f"{file_name}.txt {data}\n")

# Create subfolders in data_split folder if they don't exist
os.makedirs(test_folder, exist_ok=True)
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Split and save text files for test, train, and val folders
for file in text_files:
    file_path = os.path.join(info_folder, file)
    split_and_save_text_files(file_path, test_folder)
    split_and_save_text_files(file_path, train_folder)
    split_and_save_text_files(file_path, val_folder)
