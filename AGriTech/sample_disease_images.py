import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset", "disease"))

folders = [
    "Healthy",
    "Leaf_Blight",
    "Rust",
    "Powdery_Mildew"
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

for f in folders:
    folder_path = os.path.join(BASE, f)
    files = os.listdir(folder_path)

    image_files = [file for file in files if file.lower().endswith(IMAGE_EXTENSIONS)]

    print(f"{folder_path}: {len(image_files)} images")
