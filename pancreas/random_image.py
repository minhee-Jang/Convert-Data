# import os
# import shutil
# import random

# source_dir = "../../data/diffusion/pancrease_final_filtered/Train"

# # 라벨 별로 저장할 폴더 생성
# output_dir = "../../data/diffusion/pancrease_6000/train"
# os.makedirs(output_dir, exist_ok=True)

# # 라벨 리스트
# labels = ["BENIGN", "MALIGNANT", "NORMAL"]

# # 각 라벨 별로 선택할 이미지 수
# images_benign = 1000
# images_mal = 1000
# images_nor = 4000

# for label in labels:
#     print(label)
#     label_dir = os.path.join(output_dir, label)
#     print(label_dir)
#     os.makedirs(label_dir, exist_ok=True)
    
#     source_label_dir = os.path.join(source_dir, label)
#     image_files = os.listdir(source_label_dir)
    
#     # 이미지 파일을 무작위로 선택
#     random.shuffle(image_files)
    
#     # 이미지를 무작위로 선택하여 복사
#     if label == "BENIGN":
#         images_per_label = images_benign
#     elif label == "MALIGNANT":
#         images_per_label = images_mal
#     else:
#         images_per_label = images_nor
        
#     selected_images = random.sample(image_files, images_per_label)
    
#     for image_file in selected_images:
#         source_file = os.path.join(source_label_dir, image_file)
#         destination_file = os.path.join(label_dir, image_file)
#         shutil.copy(source_file, destination_file)

# print(f"각 라벨 별로 {images_per_label}장의 이미지를 선택 및 복사했습니다.")


import os
import shutil
import random

# Source directory
source_dir = "../../../data/diffusion/current_pancreas/pancreas_final_filtered_crop/Train"

# Output directories for train and test
output_dir_train = "../../../data/diffusion/current_pancreas/original_6000"
#output_dir_test = "../../../data/diffusion/current_pancreas/pancreas_filtered_6000/test"
os.makedirs(output_dir_train, exist_ok=True)
#os.makedirs(output_dir_test, exist_ok=True)

# Labels
labels = ["BENIGN", "MALIGNANT", "NORMAL"]

# Number of images to select for train and test
images_per_label_train = {"BENIGN": 2000, "MALIGNANT": 2000, "NORMAL": 2000}
# images_per_label_test = {"BENIGN": 300, "MALIGNANT": 300, "NORMAL": 300}

# Function to copy images
def copy_images(selected_images, source_label_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    for image_file in selected_images:
        source_file = os.path.join(source_label_dir, image_file)
        destination_file = os.path.join(output_label_dir, image_file)

        # print(source_file)
        # print(destination_file)
        shutil.copy(source_file, destination_file)

# Select and copy images to train and test folders
for label in labels:
    print(f"Processing {label}")

    source_label_dir = os.path.join(source_dir, label)
    image_files = os.listdir(source_label_dir)
    random.shuffle(image_files)

    # Select images for train and test, ensuring no overlap
    train_images = random.sample(image_files, images_per_label_train[label])
    #remaining_images = list(set(image_files) - set(train_images))
    #test_images = random.sample(remaining_images, images_per_label_test[label])

    # Copy images to train and test folders
    copy_images(train_images, source_label_dir, os.path.join(output_dir_train, label))
    #copy_images(test_images, source_label_dir, os.path.join(output_dir_test, label))

print("Image selection and copying for train and test folders completed.")