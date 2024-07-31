import numpy as np
import PIL.Image as Image
import cv2
import os
image= np.load('D:/data/diffusion/current_pancreas/fid_comparison/DIT_breast.npz')
output_dir = 'D:/data/diffusion/DIT_breast'
#mu=image['arr_0']
# print(image.shape)
array1=image['arr_0']
array2=image['arr_1']
print(len(array2))

print(array1.shape)
print(array2)
#os.mkdir('./results_image')

os.makedirs(os.path.join(output_dir, 'Lymp'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'Normal'), exist_ok=True)


folders = {
    0: 'D:/data/diffusion/DIT_breast/Lymp',
    1: 'D:/data/diffusion/DIT_breast/Normal'
}

for i in range(len(array1)):
    label = array2[i]
    folder = folders[label]
    img_path = os.path.join(folder, f'sample{i}_label{label}.png')
    cv2.imwrite(img_path, array1[i])

