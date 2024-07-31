import os
import shutil


txt_file_path = 'D:/data/diffusion/current_pancreas/2D_Pancreas_cancer_original/Test/csv/val_ser.txt'


image_directory = 'D:/data/diffusion/current_pancreas/2D_Pancreas_cancer_seg/Test/'


for cls in ['MALIGNANT', 'BENIGN', 'NORMAL']:
    os.makedirs(os.path.join(image_directory, cls), exist_ok=True)


with open(txt_file_path, 'r') as file:
    for line in file:
        file_name, file_class = line.strip().split()
        if file_class=='0':
            file_class_name='BENIGN'
        elif file_class=='1':
            file_class_name='MALIGNANT'
        else:
            file_class_name='NORMAL'
        source_path = os.path.join(image_directory, file_name)
        # print(source_path)
        # print(file_class_name)
        destination_path = os.path.join(image_directory, file_class_name, file_name)
        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)