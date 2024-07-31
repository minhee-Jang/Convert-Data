import os
import random
import shutil

# 데이터 폴더 경로 정의
data_folder = r'D:\data\CT_MAR\PNG_version'

# 각 라벨 리스트
labels = ['Mask', 'metal_trace_mask', 'recon_img_input', 'recon_img_gt', 'sinogram_input', 'sinogram_gt']

num_test_files = int(2800)
test_indices = random.sample(range(14000), num_test_files)

# 각 라벨에 대해 처리
for label in labels:
    # 라벨 폴더 경로 정의
    label_folder = os.path.join(data_folder, label)
    
    # Image와 Mask 폴더 경로 정의
    # image_folder = os.path.join(label_folder, 'images')
    #mask_folder = os.path.join(label_folder, 'masks')
    
    # 테스트 폴더 생성
    test_folder_images = os.path.join(data_folder, 'Test', label)
    #test_folder_masks = os.path.join(data_folder, 'Test', label,'masks')
    os.makedirs(test_folder_images, exist_ok=True)
    #os.makedirs(test_folder_masks, exist_ok=True)
    
    # 라벨 폴더 안의 이미지와 마스크 파일 리스트
    image_files = os.listdir(label_folder)
    #mask_files = os.listdir(mask_folder)
    image_files.sort()
 

    for idx in test_indices:
        image_file = image_files[idx]
        #mask_file = mask_files[idx]
   
        shutil.move(os.path.join(label_folder, image_file), os.path.join(test_folder_images, image_file))
        #shutil.move(os.path.join(mask_folder, mask_file), os.path.join(test_folder_masks, mask_file))