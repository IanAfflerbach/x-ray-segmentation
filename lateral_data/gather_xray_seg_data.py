import csv
import gzip
import os
from PIL import Image, ImageFilter
import shutil
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


def download_image(dicom_id, subject_id, study_id):
    source_url = 'https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/'
    patient_bucket = 'p' + subject_id[0:2]
    subject_id = 'p' + subject_id
    study_id = 's' + study_id
    img_file = dicom_id + '.jpg'

    # # get original image and downsample
    get_file(f"{source_url}{patient_bucket}/{subject_id}/{study_id}/{img_file}")
    img = Image.open(img_file)
    img = img.convert('RGB')
    img = img.resize((224, 224), Image.Resampling.NEAREST)
    img.save(img_file)
    os.replace(img_file, 'xray-lateral/' + img_file)


# get index file of x-ray data
index_filename = 'mimic-cxr-2.0.0-metadata.csv'
if not os.path.exists(index_filename):
    print("Loading X-Ray Metadata Index...")
    get_file('https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz')
    with gzip.open(index_filename + '.gz', 'rb') as f_gz:
        with open(index_filename, 'wb') as f_csv:
            shutil.copyfileobj(f_gz, f_csv)

# download datasets
TOTAL_NUM_IMAGES = 100
img_ids = []
with open(index_filename, 'r') as file:
    csv_reader = csv.DictReader(file)
    img_num = 0
    for row in csv_reader:
        if img_num >= TOTAL_NUM_IMAGES:
            break

        if row['ViewPosition'] != 'LATERAL':
            continue

        print("Image:", f"{img_num + 1}/{TOTAL_NUM_IMAGES}", "ID:", row['dicom_id'])
        download_image(row['dicom_id'], row['subject_id'], row['study_id'])
        img_ids.append(row['dicom_id'])
        img_num += 1

# write file ids
with open('index.txt', 'w') as file:
    for id in img_ids:
        file.write(f"{id}\n")