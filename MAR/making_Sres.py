from PIL import Image
import numpy as np
import os
import struct

def make_Sres(folder1, folder2, output_folder1, output_folder2):
    # 폴더 내 파일 목록 가져오기
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # 공통 파일 이름 찾기
    common_files = set(files1) & set(files2)
    print(len(common_files))

    # 공통 파일들에 대해 이미지 불러와 subtract 연산 수행
    for file_name in common_files:
        # 이미지 불러오기
        print(file_name)
        img1_p = os.path.join(folder1, file_name)
        img2_p = os.path.join(folder2, file_name)

        img1 = rawread(img1_p, [1000, 1, 900], 'float')
        img2 = rawread(img2_p, [1000, 1, 900], 'float')
        s_ma = (img1.transpose(1,0,2)).reshape(1000, 900)  # channel이 맨앞으로 가게 
        s_li = (img2.transpose(1,0,2)).reshape(1000, 900)  # channel이 맨앞으로 가게 
        # 이미지를 numpy 배열로 변환
        print(s_ma.max(), s_li.max())
        s_sub = s_ma  - s_li
        s_res = s_sub * 0.4 
        s_pre = s_res + s_li
        output_path_pre = os.path.join(output_folder1, file_name)
        s_pre.tofile(output_path_pre)

        output_path_res = os.path.join(output_folder2, file_name)
        s_res.tofile(output_path_res)
        # s_pre = (s_sub - np.min(s_sub)) / (np.min(s_sub) - np.min(s_sub)) * 255
        #s_pre = np.clip(s_pre, 0, 255).astype(np.uint8)

        # s_pre = Image.fromarray(s_sub)

        # # 차이 이미지를 저장
        # #file_name = file_name.split('.')[0]
        # print(file_name)
        # output_path = os.path.join(output_folder1, file_name)
        # s_pre.save(output_path, format='raw')

        # print(f"Processed and saved: {file_name}")

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
    if dataShape:
        data = data.reshape(dataShape)
    
    return data

def main():
    folder1 = "D:\data\CT_MAR\CT_MAR_raw_version\CT_MAR\Test\Baseline\sinogram_new_index"  # 첫 번째 폴더 경로
    folder2 = "D:\data\CT_MAR\CT_MAR_raw_version\CT_MAR\Test\LI\sinogram_new_index"  # 두 번째 폴더 경로
    output_folder1 = "D:\data\CT_MAR\CT_MAR_raw_version\CT_MAR\Test\s_pre"  # 결과를 저장할 폴더 경로
    output_folder2 = "D:\data\CT_MAR\CT_MAR_raw_version\CT_MAR\Test\s_res"
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)

    # 이미지 불러와서 subtract 연산 후 저장
    make_Sres(folder1, folder2, output_folder1, output_folder2)

if __name__ == "__main__":
    main()
