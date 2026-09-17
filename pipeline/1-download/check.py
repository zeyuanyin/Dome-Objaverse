import os

import json




# # Load the JSON file
with open('gobjaverse_280k.json', 'r') as file:
    data = json.load(file)

# # Check and print the length
# print(f"Length of the dataset: {len(data)}")


# Directory to look into
base_dir = os.environ.get('GOBJ_TAR_DIR', './gobjaverse_tar')

list_of_tar = []

# Iterate through each folder in the directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        # print(f"Folder: {folder_name}")
        
        for file_name in os.listdir(folder_path):
            # if file_name.endswith('.tar'):
            #     print(f"Tar file in {folder_name}: {file_name}")
            # if file_name.endswith(('.tar.1', '.tar.2', '.tar.3', '.tar.4', '.tar.5')):
            # if not file_name.endswith('.tar'):
            #     print(f"Matched file in {folder_name}: {file_name}")
            #     os.remove(os.path.join(folder_path, file_name))
            list_of_tar.append("/".join([folder_name, file_name.split(".")[0]]))
            
print(f"Length of the dataset: {len(list_of_tar)}")
# print(list_of_tar[:10])



# Find the non-overlapping parts between the JSON data and the tar list
json_keys = set(data)
tar_keys = set(list_of_tar)

# Keys in JSON but not in tar
json_not_in_tar = json_keys - tar_keys
print(f"Keys in JSON but not in tar: {len(json_not_in_tar)}")
print(list(json_not_in_tar)[:10])

# Keys in JSON but not in tar: 2
# ['6/41719', '23/126807']

# Keys in tar but not in JSON
tar_not_in_json = tar_keys - json_keys
print(f"Keys in tar but not in JSON: {len(tar_not_in_json)}")
print(list(tar_not_in_json)[:10])