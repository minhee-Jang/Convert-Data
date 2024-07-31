import os
import random
import shutil

# 데이터 폴더 경로 정의
data_folder = r'D:\few_shot\train'

# 각 라벨 리스트
labels = [name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))]

print(len(labels))

# 각 라벨에 대해 처리
for label in labels:
    # 라벨 폴더 경로 정의
    label_folder = os.path.join(data_folder, label, 'images')
   # print(label_folder)
    
    # Image와 Mask 폴더 경로 정의
    # image_folder = os.path.join(label_folder, 'images')
    #mask_folder = os.path.join(label_folder, 'masks')
    
    # 테스트 폴더 생성
    test_folder_images = os.path.join('D:/few_shot', 'valid', label)
    #test_folder_masks = os.path.join(data_folder, 'Test', label,'masks')
    os.makedirs(test_folder_images, exist_ok=True)
    #os.makedirs(test_folder_masks, exist_ok=True)
    
    # 라벨 폴더 안의 이미지와 마스크 파일 리스트
    image_files = os.listdir(label_folder)
    #print(image_files)
    #mask_files = os.listdir(mask_folder)
    
    # 테스트할 이미지와 마스크 파일 갯수 결정 (전체 파일의 20%)
    num_test_files = int(5)
    
    # 무작위로 테스트 파일 선택
    test_indices = random.sample(range(len(image_files)), num_test_files)
    
    # 테스트 파일들을 테스트 폴더로 이동
    for idx in test_indices:
        image_file = image_files[idx]
        #mask_file = mask_files[idx]
        
        # Image와 Mask 파일을 테스트 폴더로 이동
        shutil.move(os.path.join(label_folder, image_file), os.path.join(test_folder_images, image_file))
        #shutil.move(os.path.join(mask_folder, mask_file), os.path.join(test_folder_masks, mask_file))