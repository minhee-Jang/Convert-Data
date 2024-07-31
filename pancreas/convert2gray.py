import cv2
import os

# 변환할 이미지가 있는 폴더 경로
folder_path = "D:/data/diffusion/950_breast_uvit/Lymphadenopathy/"

# 폴더 내 모든 파일에 대해 반복
for filename in os.listdir(folder_path):
    # 파일 경로 생성
    filepath = os.path.join(folder_path, filename)
    
    # 파일이 이미지인지 확인
    if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 이미지 불러오기
        img = cv2.imread(filepath)
        
        # 이미지가 3채널인지 확인
        if img.shape[2] == 3:
            # 이미지를 흑백으로 변환
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 흑백 이미지 저장
            cv2.imwrite(filepath, gray_img)
            
            print(f"{filename} 변환 완료")
        else:
            print(f"{filename} 이미 1채널 이미지입니다. 스킵합니다.")
    else:
        print(f"{filename} 이미지 파일이 아니거나 처리할 수 없습니다. 스킵합니다.")