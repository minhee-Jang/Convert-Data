import os
import shutil
import random

# Source directory
source_dir = "E:\data\Lung_cancer\LUNG\LUNG"

# Output directories for folder 1 and folder 2
output_dir_folder1 = "E:\data\Lung_cancer\LUNG\LUNG\Test"
#output_dir_folder2 = "../../../data/diffusion/current_pancreas/original_6000/folder2"
os.makedirs(output_dir_folder1, exist_ok=True)
#os.makedirs(output_dir_folder2, exist_ok=True)

# Labels
labels = ["BENIGN", "MALIGNANT", "NORMAL"]

# Number of images to select for each folder
images_per_label = {"BENIGN": 20, "MALIGNANT": 120, "NORMAL": 82}

# Function to move images
def move_images(selected_images, source_label_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    for image_file in selected_images:
        source_file = os.path.join(source_label_dir, image_file)
        destination_file = os.path.join(output_label_dir, image_file)
        shutil.move(source_file, destination_file)

# Select and move images to folder 1 and folder 2
for label in labels:
    print(f"Processing {label}")

    source_label_dir = os.path.join(source_dir, label)
    image_files = os.listdir(source_label_dir)
    random.shuffle(image_files)

    # Divide image list into two parts
    folder1_images = image_files[:images_per_label[label]]
    # folder2_images = image_files[images_per_label[label]:]

    # Move images to folder 1
    move_images(folder1_images, source_label_dir, os.path.join(output_dir_folder1, label))

    # Move images to folder 2
    # move_images(folder2_images, source_label_dir, os.path.join(output_dir_folder2, label))

print("Image selection and moving to folder 1 and folder 2 completed.")
