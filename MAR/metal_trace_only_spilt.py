import os
import shutil

# 데이터 폴더 경로 정의
data_folder = r'D:\data\CT_MAR'

# Metal 폴더와 Training 폴더 경로 정의
metal_folder = os.path.join(data_folder, 'syndiff_recon\png')
training_folder = os.path.join('D:\data\CT_MAR\CT_MAR_PNG_version', 'Train', 'Target', 'recon_img')
metal_test_folder = os.path.join(data_folder, 'syndiff_recon', '3D_target')
os.makedirs(metal_test_folder, exist_ok=True)

# Metal 폴더와 Training 폴더의 파일 리스트 가져오기
metal_files = os.listdir(metal_folder)
training_files = os.listdir(training_folder)

# Metal 폴더의 'body'와 'head' 파일을 찾아서 Metal_test 폴더로 이동
# for metal_file in metal_files:
#     if 'body' in metal_file:
#         after_sino = metal_file.split('metalinfo')[1]
#         # 숫자 추출
#         x_number = ''.join(filter(str.isdigit, after_sino))
#        #print(x_number)
#         for training_file in training_files:
#             #print(training_file)
#             if 'body' in training_file and f'img{x_number}_' in training_file:
#                 # Pair가 되는 body 파일을 Metal_test 폴더로 이동
#                 shutil.move(os.path.join(metal_folder, metal_file), os.path.join(metal_test_folder, metal_file))
#                 print(metal_file, "같음")
#                 break
for metal_file in metal_files:
    after_sino2 = metal_file.split('sino')[1]
    # 숫자 추출
    x_number2 = after_sino2.split('_')[0]
    print(x_number2)
    print(metal_file)
    for training_file in training_files:
        if 'head' in training_file and f'img{x_number2}_' in training_file:
            # Pair가 되는 head 파일을 Metal_test 폴더로 이동
            print(training_file)
            shutil.copy(os.path.join(training_folder, training_file), os.path.join(metal_test_folder, training_file))
            break
