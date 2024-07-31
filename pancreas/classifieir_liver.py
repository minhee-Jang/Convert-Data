import os
import shutil

# 소스 폴더와 대상 폴더 경로 정의
source_folder = r'E:\data\cancer_data\NCC_Liver_Cancer_PNG'
target_folder = r'E:\data\cancer_data\NCC_Liver'

# 라벨들
labels = ['BENIGN', 'MALIGNANT', 'NORMAL']

# 각 라벨에 대해 처리
for label in labels:
    # 라벨에 해당하는 폴더 경로 정의
    label_folder = os.path.join(source_folder, label)
    
    # 해당 라벨의 ANON으로 시작하는 모든 폴더 탐색
    for root, dirs, files in os.walk(label_folder):
        for directory in dirs:
            if directory.startswith('ANON'):
                # ANON 폴더의 경로
                anon_folder = os.path.join(root, directory)
                
                # 이미지와 마스크 폴더의 경로 정의
                image_folder = os.path.join(anon_folder, 'IMAGE')
                mask_folder = os.path.join(anon_folder, 'MASK')
                
                # 대상 폴더에 images와 masks 폴더 생성
                target_label_folder = os.path.join(target_folder, label)
                os.makedirs(target_label_folder, exist_ok=True)
                target_images_folder = os.path.join(target_label_folder, 'images')
                target_masks_folder = os.path.join(target_label_folder, 'masks')
                os.makedirs(target_images_folder, exist_ok=True)
                os.makedirs(target_masks_folder, exist_ok=True)
                
                # 이미지와 마스크를 대상 폴더로 복사
                for image_file in os.listdir(image_folder):
                    # 이미지 파일을 대상 폴더에 복사
                    image_file_path = os.path.join(image_folder, image_file)
                    shutil.copy(image_file_path, target_images_folder)
                    
                    # 해당 이미지에 대한 마스크 파일도 복사
                    mask_file_path = os.path.join(mask_folder, image_file)  # 마스크 파일의 경로
                    if os.path.exists(mask_file_path):  # 마스크 파일이 존재하는 경우에만 복사
                        shutil.copy(mask_file_path, target_masks_folder)
                    else:
                        print(f"No mask found for image: {image_file_path}")
