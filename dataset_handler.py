# dataset_handler.py
# python dataset_handler.py


import os
import zipfile
import requests
import shutil

def download_and_extract_cew(destination_folder="dataset/raw_cew"):
    """
    Downloads CEW dataset zip and extracts Open/Closed images.
    """
    os.makedirs(destination_folder, exist_ok=True)
    url = "https://github.com/hysts/closed-eyes-in-the-wild/archive/refs/heads/master.zip"
    zip_path = os.path.join(destination_folder, "cew.zip")

    print("🔽 Downloading CEW dataset...")
    response = requests.get(url)
    with open(zip_path, "wb") as f:
        f.write(response.content)

    print("📂 Extracting CEW dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

    print("✅ Extraction complete.")

def organize_cew_images(raw_folder="dataset/raw_cew", target_folder="dataset/eye_states"):
    """
    Organizes extracted CEW Open/Closed eye images into target folder structure.
    """
    open_eye_dir = os.path.join(raw_folder, "closed-eyes-in-the-wild-master", "openEyes")
    closed_eye_dir = os.path.join(raw_folder, "closed-eyes-in-the-wild-master", "closedEyes")

    open_target = os.path.join(target_folder, "Open")
    closed_target = os.path.join(target_folder, "Closed")

    os.makedirs(open_target, exist_ok=True)
    os.makedirs(closed_target, exist_ok=True)

    # Copy Open eyes
    for img_name in os.listdir(open_eye_dir):
        src = os.path.join(open_eye_dir, img_name)
        dst = os.path.join(open_target, img_name)
        shutil.copyfile(src, dst)

    # Copy Closed eyes
    for img_name in os.listdir(closed_eye_dir):
        src = os.path.join(closed_eye_dir, img_name)
        dst = os.path.join(closed_target, img_name)
        shutil.copyfile(src, dst)

    print(f"✅ Organized {len(os.listdir(open_target))} Open and {len(os.listdir(closed_target))} Closed images.")

if __name__ == "__main__":
    download_and_extract_cew()
    organize_cew_images()
