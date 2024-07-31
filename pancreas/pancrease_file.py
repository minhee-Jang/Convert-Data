import numpy as np
import cv2
from matplotlib import image, pyplot as plt
from PIL import Image
import glob
import os
import shutil #파일 이동을 위한 모듈
#from skimage.metrics import structural_similarity


def make_imagelist(img_path, mask_path, result_dir):

    folderlist=os.listdir(img_path)
    print(folderlist)

    
    mask_namelist = []
    for mask_file in glob.glob(mask_path+'/*.png'):

        mask_filename = os.path.splitext(os.path.basename(mask_file))[0]  # 확장자 제거
        #print(mask_filename)
        #exit()

        
        mask_namelist.append(mask_filename)
        
    
    print("num of mask:", len(mask_namelist))
    
    for folder in folderlist:
        
        result = result_dir + folder
        os.makedirs(result, exist_ok=True)
        image_class_list = []

        print(result)
        mask_classlist = []
        for filename in glob.glob(img_path + folder + '/*.png'):
            img_file = os.path.splitext(os.path.basename(filename))[0]
            #print(img_file)
            image_class_list.append(img_file)
            # print(img_file)
            # exit()

        for mask in mask_namelist:
            #print(mask)
            if mask in image_class_list:
                #print('exist')
                mask_classlist.append(mask)
            
            #mask_classlist.extend(item for item in mask_namelist if img_file in item)
        
        print(len(mask_classlist), len(image_class_list))
        #print(mask_classlist[5])

        #before = mask_path
        #after = os.path.join(result_dir + folder)
        #print(before, after)
   
        for file in mask_classlist:
            #print(file)

            #image = os.path.basename(file)
            before = mask_path + str(file)+'.png'
            after = os.path.join(result, str(file+'.png'),)
            #print(before)
            
            if os.path.exists(before):

                #print('exist')
                shutil.copy(before, after)
            
              
        #bilateral(image_list, result_dir)
        
        print("============")

    return image_class_list, result_dir




if __name__ == "__main__" :
    
    img_path = 'D:/data/diffusion/2D_Pancreas_cancer_original/Train/'
    mask_path = 'D:/data/diffusion/2D_Pancreas_cancer_seg/Train/'
    result_dir ='D:/data/diffusion/pancrease_mask_final/Train/'

    make_imagelist(img_path, mask_path, result_dir)

    #main()



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

# def psnr_ssim(original, contrast):
#     mse = np.mean((original - contrast) ** 2)
#     if mse == 0:
#         print("MSE 0!!")
#         return 100
#     PIXEL_MAX = 255.0
#     PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#     ssim = structural_similarity(contrast, original)
#     return PSNR, ssim


# def bilateral(image_list, result_dir):
#     for i in range(len(image_list)):

        
#         #print(i, '번째 이미지')
#         img = image_list[i]
#         img= np.array(img)
#         #print(img)
#         noisy = np.uint8(np.clip(img*255,0,255))
#         dst = cv2.bilateralFilter(noisy, -1, 3, 10)
#         #dst = cv2.gausi
#         dst = np.float32(dst/255)
#         #print(dst)
#         image =Image.fromarray(dst)
#         image.save(result_dir+'/'+'{:03d}'.format(i)+'.tiff','TIFF')
