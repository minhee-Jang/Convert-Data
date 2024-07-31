import numpy as np
import cv2
from matplotlib import image, pyplot as plt
from PIL import Image
import glob
import os
import math
from skimage.metrics import structural_similarity


def make_imagelist(path, result_dir,frame):

    folderlist=os.listdir(path)
    print(folderlist)
    
    for folder in folderlist:
        image_list = []
        print(folder)
        for filename in glob.glob(path+folder+'/*.tiff'):
            im=Image.open(filename)
            image_list.append(im)
        print("num of image:",len(image_list))    #num of image in folder
        result_dir = result_dir+folder
        os.makedirs(result_dir, exist_ok=True)
        print(result_dir)
        nlm(image_list, result_dir)                   ####filter 변경
        
        
        print("============")

    return image_list, result_dir

def nlm(image_list, result_dir):

    h = 4
    templateWS = 7
    searchWS= 21

    for i in range(len(image_list)):
        
        #print(i, '번째 이미지')
        img = image_list[i]
        img= np.array(img)
        #print(img)
        print(img.shape)
        img = np.uint8(np.clip(img*255,0,255))
        dst = cv2.fastNlMeansDenoising(img, None ,h, templateWS, searchWS)
        dst = np.float32(dst/255)
        nlm_image =Image.fromarray(dst) 
        nlm_image.save(result_dir+'/'+'{:03d}'.format(i)+'.tiff','TIFF')

def bilateral(image_list, result_dir):
    for i in range(len(image_list)):

        
        #print(i, '번째 이미지')
        img = image_list[i]
        img= np.array(img)
        #print(img)
        noisy = np.uint8(np.clip(img*255,0,255))
        dst = cv2.bilateralFilter(noisy, -1, 3, 10)
        #dst = cv2.gausi
        dst = np.float32(dst/255)
        #print(dst)
        image =Image.fromarray(dst)
        image.save(result_dir+'/'+'{:03d}'.format(i)+'.tiff','TIFF')


# def psnr_ssim(original, contrast):
#     mse = np.mean((original - contrast) ** 2)
#     if mse == 0:
#         print("MSE 0!!")
#         return 100
#     PIXEL_MAX = 255.0
#     PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#     ssim = structural_similarity(contrast, original)
#     return PSNR, ssim

# def main():
#     dir_path = 'D:/data/video-sr/moving700/test/low'
#     nlm_list = []
#     for filename in glob.glob(dir_path+'/*.tiff'):
#             im=Image.open(filename)
#             nlm_list.append(im)
#     print(len(nlm_list))     

#     hr_dir = 'D:/data/denoising/fasunet_0_1'
#     hr=[]
#     for filename in glob.glob(hr_dir+'/*.tiff'):
#             im=Image.open(filename)
#             hr.append(im)
#     print(len(hr))

    
#     avg = 0
    
#     for i in range(len(nlm_list)):
#         #print(i)
#         contrast = np.array(nlm_list[i])
#         contrast = np.uint8(contrast*255)
#         #print(contrast)
#         original = np.array(hr[i+2])
#         original = np.uint8(original*255)
#         print(original)
#         psnr, ssim = psnr_ssim(original, contrast)
#         avg_psnr+=psnr
#         avg_ssim+=ssim
#         print(f"PSNR value of frame{i+2} is {psnr} dB")
#         print(f"PSNR value of frame{i+2} is {ssim} dB")
    
#     avg_psnr = avg_psnr/len(nlm_list)
#     avg_ssim = avg_ssim/len(nlm_list)
#     print("avg_psnr", avg_psnr)
#     print("avg_ssim", avg_ssim)





if __name__ == "__main__" :
    
    path = 'D:/data/mayo2d/mayo2d/test/quarter_1mm/'
    result_dir ='D:/data/nlm_231005_mayo2d/'
    frame=3
    make_imagelist(path, result_dir, frame)

    #main()