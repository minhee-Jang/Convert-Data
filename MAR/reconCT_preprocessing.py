import shutil
from PIL import Image
import os
import numpy as np
import struct

def convert_raw_to_png(input_folder, output_folder):
    # 입력 폴더 내의 모든 파일에 대해 반복
    for filename in os.listdir(input_folder):
        #if filename.endswith('.raw'):
        if filename.endswith('.raw'):
            # .raw 확장자 파일을 읽어서 PNG로 변환
            #after_sino = filename.split('body_')[1]
            #x_number = after_sino.split('_')[0]
            #new_filename = f"body_{x_number}"
            raw_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.split('.raw')[0] + '.png')

            with open(raw_path, 'rb') as raw_file:
                # .raw 파일을 열어서 Pillow Image로 로드
                image = rawread(raw_path, [512, 1, 512], 'float')
                #image = image.reshape(900, 1000)
                image = (image.transpose(1,0,2)).reshape(512, 512)  # channel이 맨앞으로 가게 
                # print(np.min(image))
                # histogram, bin_edges = np.histogram(image, bins=256, range=(np.min(image), np.max(image) + 1))
                # cdf = histogram.cumsum()
                # cdf_normalized = cdf / float(cdf.max())  # CDF를 정규화
                # img_equalized = np.interp(image.flatten(), xp=np.arange(256), fp=cdf_normalized * 255)
                # img_equalized = np.reshape(img_equalized, image.shape)
                # img_equalized = Image.fromarray(img_equalized.astype(np.uint8), mode='L')

                p1, p99 = np.percentile(image, (0.1, 99.9))  # 1%와 99% 백분위수
                image_clipped = np.clip(image, p1, p99)

                image_rescaled = (image_clipped - p1) / (p99 - p1) * 255
                image_rescaled = image_rescaled.astype(np.uint8)  # uint8 타입으로 변환

                #image_scaled = (((image - np.min(image)) / (np.max(image) - np.min(image))))
                # PNG로 저장
                print(np.min(image_rescaled), np.max(image_rescaled) )
                img = Image.fromarray(image_rescaled)
                # image_scaled = Image.fromarray((image_scaled * 255).astype(np.uint8), mode='L')
                img.save(output_path, format='PNG')
            print(f"{filename} 변환 완료")

        # if filename.endswith('512x512x1.raw'):
        #     raw_path = os.path.join(input_folder, filename)
        #     output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

        #     # PNG 파일을 열어서 Pillow Image로 로드
        #     image = Image.open(raw_path)
        #     # 이미지를 numpy 배열로 변환하여 0과 1 사이의 값으로 스케일링
        #     #binary_mask = np.array(image) / 255.0

        #     # 0에서 255 사이의 값으로 변환
        #     image_array = np.array(image)
        #     print(np.max(image_array))
        #     #mask_uint8 = (image_array * 255).astype(np.uint8)
        #     #print(np.max(mask_uint8))

        #     # 변환된 값으로 이미지 생성
        #     mask_image = Image.fromarray(image_array)

        #     # PNG로 저장
        #     mask_image.save(output_path, format='PNG')
        #     print(f"{filename} 변환 완료")



def organize_files(source_dir):

    files = os.listdir(source_dir)
    

    sino_dir = os.path.join(source_dir, 'sinogram')
    os.makedirs(sino_dir, exist_ok=True)
    

    img_dir = os.path.join(source_dir, 'recon_img')
    os.makedirs(img_dir, exist_ok=True)
    
    # 파일들을 식별하고 적절한 폴더로 이동합니다.
    for file in files:
        if 'sino' in file:
            shutil.move(os.path.join(source_dir, file), sino_dir)
        elif 'img' in file:
            shutil.move(os.path.join(source_dir, file), img_dir)

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

def sino_normalization(path):
    img = rawread(path, [1000, 900, 1], 'float')
    img = img.transpose(2,1,0)  # channel이 맨앞으로 가게 
    img_scaled = (((img - np.min(img)) / (np.max(img) - np.min(img))))
    return img_scaled

if __name__ == "__main__":
    source_directory = 'D:\data\CT_MAR\Target'  # 기존 파일들이 있는 폴더명을 여기에 넣으세요.
    # 입력 폴더와 출력 폴더를 지정합니다.
    input_folder = 'D:\data\CT_MAR\phase2_feedback_test_data-selected\PNG\Xsino_recon'
    output_folder = 'D:\data\CT_MAR\phase2_feedback_test_data-selected\PNG\Xsino_recon'
    convert_raw_to_png(input_folder, output_folder)

    #organize_files(source_directory)
