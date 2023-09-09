import json

# Path to the JSON file
json_file_path = r'C:\Project\new D\validation\validation\annos\000001.json'

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

# Now you can work with the json_data dictionary
print(json_data)
