import os
import tarfile
from tqdm import tqdm
import random

base_dir = os.environ.get('GOBJ_TAR_DIR', './gobjaverse_tar')

target_dir = os.environ.get('GOBJ_UNZIP_DIR', './gobjaverse_unzip')

list_of_tar = []

# Iterate through each folder in the directory

folders = os.listdir(base_dir)
random.shuffle(folders)

# folders = ['118', '119', '120', '121', '122', '123', '124', '125', '126', '127']

# for folder_name in tqdm(os.listdir(base_dir)):
for folder_name in tqdm(folders):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        
        print(os.path.join(target_dir, folder_name))
        # os.makedirs(os.path.join(target_dir, folder_name))
        # if os.path.exists(os.path.join(target_dir, folder_name)):
        #     continue
        
        for file_name in tqdm(os.listdir(folder_path)):
            if file_name.endswith('.tar'):
                list_of_tar.append("/".join([folder_name, file_name]))
                
                # Extract the tar file into the corresponding folder in the target directory
                tar_path = os.path.join(folder_path, file_name)
                extract_path = os.path.join(target_dir, folder_name)
                extract_done = os.path.join(extract_path, file_name.split(".")[0])
                if os.path.exists(extract_done):
                    extract_view_path = os.path.join(extract_done, "campos_512_v4")
                    folders_in_view = [d for d in os.listdir(extract_view_path)]
                    if len(folders_in_view) == 40:
                        continue
                    
                
                print(f"Extracting {tar_path} to {extract_path}")
                with tarfile.open(tar_path, 'r') as tar:
                    tar.extractall(path=extract_path)
                    
                # exit()
                
                
print(f"Length of the dataset: {len(list_of_tar)}")
