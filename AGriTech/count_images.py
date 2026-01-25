import os

base_path = 'd:/AgriTechProject/AGriTech/dataset/disease'

folders = ['Healthy', 'Leaf_blight', 'Rust', 'Powdery_Mildew']

for folder in folders:
    path = os.path.join(base_path, folder)
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.jpg'):
                all_files.append(os.path.join(root, file))
    
    all_files.sort()  # sort by path/name
    if len(all_files) > 250:
        to_delete = all_files[250:]
        for f in to_delete:
            os.remove(f)
        print(f"{folder}: deleted {len(to_delete)} images, kept 250")
    else:
        print(f"{folder}: already {len(all_files)} images (<=250)")