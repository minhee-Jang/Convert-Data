import numpy as np
import cv2
from matplotlib import image, pyplot as plt
from PIL import Image
import glob
import os
import shutil #파일 이동을 위한 모듈
#from skimage.metrics import structural_similarity


def make_imagelist(img_path, mask_path, result_dir):

    folderlist_img=os.listdir(img_path)
    print(folderlist_img)
    folderlist_mask=os.listdir(mask_path)
    print(folderlist_mask)

    
  
    for folder in folderlist_img:

        output_directory = result_dir + folder

        input_folder = img_path + folder
        mask_folder = mask_path + folder

        input_files = sorted([f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))])
        mask_files = sorted([f for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f))])
        #print(input_files[0])

        for input_file, mask_file in zip(input_files, mask_files):
            # input_file_name_core = os.path.splitext(input_file)[0]
            # mask_file_name_core = os.path.splitext(mask_file)[0].replace('_seg', '')  # '_seg'를 제거하여 원본 파일 이름만 남깁니다.
        
        #if input_file_name_core == mask_file_name_core:
            
            mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
            mask = mask
            count_ones = np.sum(mask == 255)
    
            if count_ones >= 500:
                original_img = cv2.imread(os.path.join(input_folder, input_file))

                y_coords, x_coords = np.where(mask == 255)
                y_mean = int(np.mean(y_coords))
                x_mean = int(np.mean(x_coords))
              
                patch_size = 256  
                start_x = int(x_mean - patch_size / 2)
                start_y = int(y_mean - patch_size / 2)


                start_x = max(0, min(start_x, original_img.shape[1] - patch_size))
                start_y = max(0, min(start_y, original_img.shape[0] - patch_size))

                patch_original = original_img[start_y:start_y + patch_size, start_x:start_x + patch_size]
                patch_mask = mask[start_y:start_y+patch_size, start_x:start_x+patch_size]

            
                if not os.path.exists(output_directory):
                    result_out = output_directory
                    os.makedirs(result_out)
                    #mask_out = os.path.join('D:/data/diffusion/current_pancreas/pancrease_mask_256_ver2/Train/', folder)
                    #os.makedirs(mask_out)

                cv2.imwrite(os.path.join(result_out, input_file), patch_original)
                #cv2.imwrite(os.path.join(mask_out, mask_file), patch_mask)
        print("============")


if __name__ == "__main__" :
    
    img_path = 'D:/data/diffusion/current_pancreas/pancreas_final_filtered/Test/'
    mask_path = 'D:/data/diffusion/current_pancreas/pancreas_final_filtered_mask/Test/'
    result_dir ='D:/data/diffusion/current_pancreas/pancreas_final_filtered_crop/Test/'

    make_imagelist(img_path, mask_path, result_dir)



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