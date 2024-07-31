import pandas as pd
import os
import shutil

# Replace these paths with the actual paths on your system
csv_file_path = 'D:/data/APTOS2019/aptos2019-blindness-detection/test.csv'
images_directory = 'D:/data/APTOS2019/aptos2019-blindness-detection/test_images/'
output_directory = 'D:/data/APTOS2019/aptos2019-blindness-detection/test_images/'

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Create directories for each class if they don't exist
for class_id in range(5):  # Assuming class ids are 0, 1, 2, 3, 4
    class_folder = os.path.join(output_directory, str(class_id))
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

# Move each image to the corresponding class directory
for index, row in data.iterrows():
    image_file = row['id_code'] + '.png'  # Assuming images are in png format
    source_path = os.path.join(images_directory, image_file)
    destination_path = os.path.join(output_directory, str(row['diagnosis']), image_file)
    print(index)
    print(row['id_code'])
    # print(image_file)
    #print(source_path)

    # print(destination_path)
    # exit(0)
    if os.path.exists(source_path):  # Check if the source image exists
        shutil.move(source_path, destination_path)
    else:
        print(f"Image {image_file} not found in {images_directory}")

print("Images have been sorted into respective folders.")