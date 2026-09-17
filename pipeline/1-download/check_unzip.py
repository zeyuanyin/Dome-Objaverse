import os
import tqdm

target_dir = os.environ.get('GOBJ_UNZIP_DIR', './gobjaverse_unzip')


folders = os.listdir(target_dir)


uncomplete_folders = []

for folder_name in tqdm.tqdm(folders): # 159
    folder_path = os.path.join(target_dir, folder_name)  
    for file_name in tqdm.tqdm(os.listdir(folder_path)): #808530
        view_path = os.path.join(folder_path, file_name, 'campos_512_v4')
        subfolders = [f for f in os.listdir(view_path) if os.path.isdir(os.path.join(view_path, f))]
        if len(subfolders) != 40:
            uncomplete_folders.append(view_path)


with open('uncomplete_folders.txt', 'w') as f:
    for folder in uncomplete_folders:
        print(folder)
        f.write(folder + '\n')
