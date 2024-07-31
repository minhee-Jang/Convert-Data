import numpy as np
import cv2
from matplotlib import image, pyplot as plt
from PIL import Image
import glob
import os
import shutil 

def make_imagelist(img_path, mask_path, result_dir1, result_dir2):
    folderlist_img = os.listdir(img_path)
    folderlist_img.remove('csv')
    print(folderlist_img)
    
    folderlist_mask = os.listdir(mask_path)
    print(folderlist_mask)

    for folder in folderlist_img:
        output_directory = result_dir1 + folder

        input_folder = img_path + folder
        mask_folder = mask_path + folder

        input_files = sorted([f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))])
        mask_files = sorted([f for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f))])

        for input_file, mask_file in zip(input_files, mask_files):
            mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
            #mask = mask / 255
            count_ones = np.sum(mask == 255)

            if count_ones >= 1000:  # count_ones가 1000 이상인 경우에만 저장
                original_img = cv2.imread(os.path.join(input_folder, input_file))
                result_out = output_directory
                if not os.path.exists(output_directory):
                    os.makedirs(result_out)

                original_out = os.path.join(result_dir2, folder)  # 마스크 저장 경로 수정
                if not os.path.exists(original_out):
                    os.makedirs(original_out)

                cv2.imwrite(os.path.join(result_out, input_file), mask )
                cv2.imwrite(os.path.join(original_out, mask_file), original_img)

        print("============")

if __name__ == "__main__":
    img_path = 'D:/data/diffusion/current_pancreas/2D_Pancreas_cancer_original/Test/'
    mask_path = 'D:/data/diffusion/current_pancreas/2D_Pancreas_cancer_seg/Test/'
    result_dir1 = 'D:/data/diffusion/current_pancreas/pancreas_final_filtered_mask/Test/'
    result_dir2 = 'D:/data/diffusion/current_pancreas/pancreas_final_filtered/Test/'

    make_imagelist(img_path, mask_path, result_dir1, result_dir2)