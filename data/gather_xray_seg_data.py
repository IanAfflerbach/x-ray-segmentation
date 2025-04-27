import csv
import os
from PIL import Image, ImageFilter
import subprocess


def get_file(filepath):
    username='PHYSIONET-USERNAME'
    password='PHYSIONET-PASSWORD'
    command = [
            'wget',
            f'--user={username}',
            f'--password={password}',
            filepath
        ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def download_images(img_name, img_path):
    segmented_url = 'https://physionet.org/files/chest-x-ray-segmentation/1.0.0/'
    original_url = 'https://physionet.org/files/mimic-cxr-jpg/2.1.0/'

    # get segmented images
    segmented_path = img_path
    mask_path = img_path.replace('.jpg', '-mask.jpg')

    get_file(segmented_url + segmented_path)
    os.replace(img_name + '.jpg', 'xray-segmented/' + img_name + '.jpg')
    get_file(segmented_url + mask_path)
    os.replace(img_name + '-mask.jpg', 'xray-segment-mask/' + img_name + '-mask.jpg')

    # get original image and downsample
    get_file(original_url + img_path)
    img = Image.open(img_name + '.jpg')
    img = img.convert('RGB')
    img = img.resize((224, 224), Image.Resampling.NEAREST)
    img.save(img_name + '.jpg')
    os.replace(img_name + '.jpg', 'xray-raw/' + img_name + '.jpg')


# get index file of segmented data
index_filename = 'CXLSeg-segmented.csv'
if not os.path.exists(index_filename):
    print("Loading Segmentation Index...")
    get_file('https://physionet.org/files/chest-x-ray-segmentation/1.0.0/CXLSeg-segmented.csv')

# download datasets
TOTAL_NUM_IMAGES = 1000
img_ids = []
with open(index_filename, 'r') as file:
    csv_reader = csv.DictReader(file)
    img_num = 0
    for row in csv_reader:
        if img_num >= TOTAL_NUM_IMAGES:
            break
        print("Image:", f"{img_num + 1}/{TOTAL_NUM_IMAGES}", "ID:", row['dicom_id'])
        download_images(row['dicom_id'], row['DicomPath'])
        img_ids.append(row['dicom_id'])
        img_num += 1

# write file ids
with open('index.txt', 'w') as file:
    for id in img_ids:
        file.write(f"{id}\n")