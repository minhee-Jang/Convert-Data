
"""
0. 설치 필요 라이브러리 (Windows 기준 터미널 명령어)
    (1) pip install opencv-python
    (2) pip install numpy
    (3) pip install Pillow
​
1. 공통 사항
    (1) 파라미터 h: Parameter regulating filter strength. 
    Big h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise
    (2) 파라미터 templateWS: Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels
    (3) 파라미터 searchWS: Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels.
    (4) 파라미터 average: 아래 2.와 3.에서 설명
    (5) nlm 결과 이미지 명명 규칙은 '해당폴더이름_n_frame수_(h, templateWS, searchWS)_frame넘버.tiff'임
​
2. average filter 뽑은 후 그 이미지로 nlm           #bm3d버전 
    (1) 불러오는 이미지 경로: './genoray/low/DIRECTORY_NAME/'
        (예시) './genoray/low/Spine_AP_mA-10/000.tiff'
    (2) 파라미터 값 설정: n_frame = 5 (5장 평균 (ex. n_frame =1 : signle input))
    (3) Average 결과 이미지 저장 경로: './genoray/save/avged'
        (예시) './genoray/save/avged/Spine_AP_mA-10/Spine_AP_mA-10_5_000_averaged.tiff'
    (4) BM3D 결과 이미지 저장 경로: './genoray/save/bm3ded'
        (예시) './genoray/save/bm3ded/Spine_AP_mA-10/Spine_AP_mA-10_5_0.05_000_bm3ded.tiff'
​
******************************
    minhee Jang
    MBE, Ewha Womans Univ.
    mh33445@ewhain.net
******************************
"""
import numpy as np
import cv2
from PIL import Image
import glob
import os

##Apply nlm
def nlmfilter2d(image_list, result_folder,n_frame, h, templateWS, searchWS):

    #url_name = 'n{}m{}g{}n{}'.format(opt.n_inputs, opt.ms_channels, opt.growth_rate, opt.n_denselayers)
    #parameter = n_frame+"_("+str(h)+','+str(templateWS)+','+str(searchWS)+')'
    parameter = str(n_frame)+'_h{}_templateWS{}_searchWS{}'.format(h, templateWS, searchWS)
    result_dir = os.path.join(result_folder ,parameter)
    os.makedirs(result_dir, exist_ok=True)

    for i in range(len(image_list)-n_frame+1):
        #print('index=',i)
        img=[]    
        total=0
        ###get frames average
        for j in range(i,i+n_frame):
           img.append(image_list[j])
        print("i th=", i)               

        noisy = [np.float64(i) for i in img]        #frame average single input
        for k in range(n_frame):
            total += noisy[k]
        noisy = total /n_frame    

        noisy = np.uint8(np.clip(noisy*255,0,255))
        dst = cv2.fastNlMeansDenoising(noisy, None ,h, templateWS, searchWS)
        dst = np.float32(dst/255)
        nlm_image =Image.fromarray(dst) 
        nlm_image.save(result_dir+'\\'+folder+'_{:03d}'.format(i)+'.tiff','TIFF')
        print(i, 'th nlm_image finished')



if __name__ == "__main__" :
    
    data_path = 'D:/data/denoising/genoray/train/low/'
    result_dir ='D:/data/denoising/genoray/result/avg_5_nlm/'

    #Set Parameter 
    n_frame=5
    h = 4
    templateWS = 7
    searchWS= 21


    folderlist=os.listdir(data_path)
    print(folderlist)
    

    for folder in folderlist:
        image_list = []
        print(folder)
        for filename in glob.glob(data_path+folder+'/*.tiff'):
            im=Image.open(filename)
            image_list.append(im)
        print("num of image:",len(image_list))    #num of image in folder
        result_folder = result_dir + folder
        os.makedirs(result_dir, exist_ok=True)
        print(result_dir)
        nlmfilter2d(image_list, result_folder, n_frame, h, templateWS, searchWS)
        
        print("============folder change================")
