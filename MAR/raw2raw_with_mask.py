import numpy as np
import os
import glob
import natsort
import cv2
import re
import struct
png_dir =r'D:\data\CT_MAR\Phase3_test\Real_Final_phase3_results\final\raw\formetric'
raw_dir =r'D:\data\CT_MAR\Phase3_test\Real_Final_phase3_results\final\final_raw\final_raw_scaling\formetric'
mask_dir =r'D:\data\CT_MAR\Phase3_test\\phase3_final_test_data\baseline\recon_img'

def rawwrite(fname, data):
    with open(fname, 'wb') as fout:
        fout.write(data)

def rawread(fname, dataShape, dataType):
    # dataType is for numpy, ONLY allows: 'float'/'single', 'double', 'int'/'int32', 'uint'/'uint32', 'int8', 'int16'
    #          they are single, double, int32, uin32, int8, int16
    with open(fname, 'rb') as fin:
        data = fin.read()
    # https://docs.python.org/3/library/struct.html
    switcher = {'float': ['f', 4, np.single],
                'single': ['f', 4, np.single],
                'double': ['d', 8, np.double],
                'int': ['i', 4, np.int32],
                'uint': ['I', 4, np.uint32],
                'int32': ['i', 4, np.int32],
                'uint32': ['I', 4, np.uint32],
                'int8': ['b', 1, np.int8],
                'int16': ['h', 2, np.int16]}
    fmt = switcher[dataType]
    data = struct.unpack("%d%s" % (len(data)/fmt[1], fmt[0]), data) # interpret bytes as packed binary data
    data = np.array(data, dtype=fmt[2])
    #print(data.shape)
    if dataShape:
        data = data.reshape(dataShape)
    return data

mask = sorted(os.listdir(mask_dir))
files = sorted(os.listdir(raw_dir))
gt_max = 1000
gt_min = -1000
for i, filename in enumerate(files, 1):
    #print(filename)
    if filename.endswith('.raw'):
        file =os.path.join(raw_dir, filename)
        print(file)
        after_sino = filename.split('.')[0]
        # print(after_sino)
        # exit()
        mask_path = os.path.join(mask_dir, f'body_{after_sino}.png')
        raw =os.path.join(raw_dir , filename)
        print(mask_path)
        mask = np.array(cv2.imread(mask_path ,cv2.IMREAD_GRAYSCALE))
        image = rawread(raw, [512, 1, 512], 'float')
        mask_binary = np.where(mask >= 130.0, 1, 0)
        mask_binary = np.expand_dims(mask_binary, axis=1)
        max_value = 1851
        print(max_value)
        print(image.shape, mask_binary.shape)
        image[mask_binary == 1] = max_value# 
        #extracted_number = re.findall(r'\d+', f.split('\\')[-1])[0]
       #  png_img = np.array(cv2.imread(raw ,cv2.IMREAD_GRAYSCALE)) /255.0 *(gt_max - gt_min) + gt_min
        # png_img = image /255.0 *(gt_max - gt_min) + gt_min
        #print(png_img.shape)
        png_img = image.astype(np.float32)
        print(png_img.shape)
        print(f"Image shape: {png_img.shape}, Data type: {png_img.dtype}")
        png_tobytes=png_img.tobytes()
        print(f"Size of png_tobytes: {len(png_tobytes)} bytes")   
        rawwrite(f'D:/data/CT_MAR/Phase3_test/Real_Final_phase3_results/final/final_raw/final_raw_scaling_with_mask_0526/{after_sino}.raw', png_tobytes)

#     #img=rawread(f'./raw_img_new/{extracted_number}.raw',[1000,900,1], 'float')
#     #exit()
# for f in natsort.natsorted(glob.glob(png_dir)):
#     print(f)

#     extracted_number = re.findall(r'\d+', f.split('\\')[-1])[0]
#     print(extracted_number)
#     exit()
#     png_img=np.array(cv2.imread(f,cv2.IMREAD_GRAYSCALE))
#     png_img = png_img.astype(np.float32)
#     print(f"Image shape: {png_img.shape}, Data type: {png_img.dtype}")
#     png_tobytes=png_img.tobytes()
#     print(f"Size of png_tobytes: {len(png_tobytes)} bytes")
#     raw_img=rawwrite(f'E:/MAR/raw_float_img/{extracted_number}.raw',png_tobytes)