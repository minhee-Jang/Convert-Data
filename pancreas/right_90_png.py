from PIL import Image
import os
import shutil

# # 이미지가 있는 폴더 경로
# folder_path = 'E:/data/cancer_data/Liver_Cancer_dataset/Training/Normal/'
# save_path = 'E:/data/cancer_data/Liver_Cancer_dataset/Training/Normal_rotate/'
# os.makedirs(save_path)

# # 폴더 내 모든 파일에 대해 반복
# for filename in os.listdir(folder_path):
#     if filename.endswith('.png'):
#         image_path = os.path.join(folder_path, filename)
#         image = Image.open(image_path)
#         rotated_image = image.rotate(180)

#         rotated_image.save(os.path.join(save_path, filename))


classes = {'0': 'class_0', '1': 'class_1'}

# 폴더 생성
for class_name in classes.values():
    os.makedirs(os.path.join('E:/data/cancer_data/MRI_Breast_Cancer/Test', class_name), exist_ok=True)

# 파일 이동
file_class_mapping = {}  # 파일명과 클래스 매핑을 저장할 딕셔너리

# val_set.txt 파일 읽기
with open('E:/data/cancer_data/MRI_Breast_Cancer/Test/csv/val_ser.txt', 'r') as file:
    for line in file:
        file_name, class_name = line.strip().split()  # 파일명과 클래스를 공백을 기준으로 분리
        file_class_mapping[file_name] = class_name

# 파일 이동
for file_name, class_name in file_class_mapping.items():
    src = os.path.join('E:/data/cancer_data/MRI_Breast_Cancer/Test/', file_name)
    dest = os.path.join('E:/data/cancer_data/MRI_Breast_Cancer/Test/', classes[class_name], file_name)
    shutil.move(src, dest)