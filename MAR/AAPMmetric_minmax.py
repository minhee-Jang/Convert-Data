import numpy as np
import os
from tqdm import tqdm
import sys
import gecatsim as xc
from skimage.metrics import structural_similarity as SKssim
from skimage.metrics import peak_signal_noise_ratio as SKpsnr
import glob
from PIL import Image

import gecatsim as xc
from gecatsim.pyfiles.CommonTools import *

rootpath = "D:/data/CT_MAR/Phase3_test/Dannet_syndiff_results"

# ground truth path
GT_root = os.path.join(rootpath, 'gt_body')
# MAR output path
my_root = 'D:\data\CT_MAR\Phase3_test\Real_Final_phase3_results\final\raw\formetric'
# metal mask path
mask_root = os.path.join(rootpath, 'mask_body' )

def rawwrite(fname, data):
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)

    with open(fname, 'wb') as fout:
        fout.write(data)

def get_ring_mask():
    #mask = np.zeros((512, 512), dtype=int)
    mask_diam = 470 # in mm
    mask_diam_pix = mask_diam/(400/512) # in pixels
    indice = np.indices((512, 512))
    mask = (indice[0]-255.5)**2 + (indice[1]-255.5)**2 < mask_diam_pix**2/4.
    return mask

allpaths = glob.glob(GT_root+"/*512x512x1.raw")
print(allpaths[0])

FOVmask = get_ring_mask()

ssim_list, psnr_list = [], []
out_list = []
myimg_paths = []
rmse_list = []

gt_min_list = []
gt_max_list = []

for thissub in tqdm(allpaths):
    path_split = thissub.split('/')
    #print(path_split)
    #print('here')
    x_num = (path_split[-1].split("img")[1]).split('_')[0]
    #print(x_num)
    #exit()
    # check difference
    mypath_all = glob.glob(os.path.join(my_root, f'body_{x_num}.raw'))
    #print(mypath_all)
    # mypath_all = glob.glob(my_root+"/*_"+path_split[-1].split('_')[3]+".*")
    if len(mypath_all) != 1:
        print("ERROR! Wrong number of MAR results found.\n")
        sys.exit(1)
    mypath = mypath_all[0]
    myimg_paths.append(mypath)
    myrecon = xc.rawread(mypath, [512, 512, 1], 'float')
    gt_path = thissub
    gtrecon = xc.rawread(gt_path, [512, 512, 1], 'float')
    # print(type(gtrecon))
    # exit()
    p1, p99 = np.percentile(gtrecon, (0.1, 99.9))
    # print(p1, p99)
    # exit()
    gt_min = -1000
    gt_max = 2500
    myrecon = myrecon / 255.0 * (gt_max - gt_min) + gt_min
    # print(myrecon.min(), myrecon.max())

    # myrecon = (myrecon - myrecon.min()) / (myrecon.max() - myrecon.min()) * 1000
    myrecon += 1000

    
    
    # gtrecon = (gtrecon - gtrecon.min()) / (gtrecon.max() - gtrecon.min()) * 1000
    # gt_min_list.append(gtrecon.min())
    # gt_max_list.append(gtrecon.max())
    # print(gtrecon.min(), gtrecon.max())
    gtrecon += 1000

# print(sum(gt_min_list)/len(allpaths))
# print(sum(gt_max_list)/len(allpaths))

    # metal mask (ground truth)
    gt_metal_mask_path = os.path.join(mask_root, f'body_{x_num}.raw')

    # 경로에 맞게 수정 
    # parts = gt_metal_mask_path.split('/')
    # parts = [part for part in parts if part != 'recon_img']
    # gt_metal_mask_path = '/'.join(parts)

    gt_metal_mask = xc.rawread(gt_metal_mask_path, [512, 512, 1], 'float')
    gt_metal_mask = gt_metal_mask>0.5

    # calc PSNR
    myrecon[gt_metal_mask] = 0
    gtrecon[gt_metal_mask] = 0
    min2 = -2000
    max2 = 6000
    gtrecon = (gtrecon-min2)/(max2-min2)
    myrecon = (myrecon-min2)/(max2-min2)
    gtrecon = np.clip(gtrecon, 0, 1)[:,:,0]
    myrecon = np.clip(myrecon, 0, 1)[:,:,0]
    # only body has max FOV limit
    if 'body' in thissub:
        myrecon[FOVmask<0.5] = 0
        gtrecon[FOVmask<0.5] = 0
    
    # Visualizatoin 
    # save_img = Image.fromarray(myrecon.astype('uint8'), mode='L')
    # save_img.save('/mnt/aix21104/jw/Syndiff/test/myrecon.png')
    # rawwrite('/mnt/aix21104/jw/Syndiff/test/myrecon.raw', myrecon)

    # save_img = Image.fromarray(gtrecon.astype('uint8'), mode='L')
    # save_img.save('/mnt/aix21104/jw/Syndiff/test/gtrecon.png')
    # rawwrite('/mnt/aix21104/jw/Syndiff/test/gtrecon.raw', gtrecon)
    # exit()

    if my_root == GT_root:
        psnr = SKpsnr(gtrecon, myrecon-0.001, data_range=1)
    else:
        psnr = SKpsnr(gtrecon, myrecon, data_range=1)
    # calc SSIM
    if my_root == GT_root:
        ssim = SKssim(gtrecon, myrecon-0.001, win_size=11, data_range=1, gaussian_weights=True)
    else:
        ssim = SKssim(gtrecon, myrecon, win_size=11, data_range=1, gaussian_weights=True)
    rmse = np.sqrt(np.mean((gtrecon-myrecon)**2))

    ssim_list.append(ssim)
    psnr_list.append(psnr)
    rmse_list.append(rmse)
    out_list.append([ssim, psnr])


results_name = 'results_head_real_minamx_body.txt'
with open(results_name, 'w') as f:
    f.write("# rmse, ssim, psnr, gt_path, my_path\n")
    for i in range(len(psnr_list)):
        f.write("{:10.6f} {:10.6f} {:10.6f} {} {}\n".format(rmse_list[i], ssim_list[i], psnr_list[i], allpaths[i], myimg_paths[i]))

print(sum(rmse_list)/len(psnr_list), '/', sum(ssim_list)/len(psnr_list), '/', sum(psnr_list)/len(psnr_list))