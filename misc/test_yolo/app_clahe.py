!pip install ultralytics opencv-python matplotlib
!pip install -U ultralytics


import os
import cv2
from tqdm import tqdm

#parametros do CLAHE
clipLimit    = 2.0
tileGridSize = (8, 8)

SRC_ROOT  = '/content/content/thermal'
DST_ROOT  = '/content/content/thermal_proc'

for split in ['train', 'val']:
    src_imgs = os.path.join(SRC_ROOT,  split, 'images')
    src_lbls = os.path.join(SRC_ROOT,  split, 'labels')
    dst_imgs = os.path.join(DST_ROOT,  split, 'images')
    dst_lbls = os.path.join(DST_ROOT,  split, 'labels')
    os.makedirs(dst_imgs, exist_ok=True)
    os.makedirs(dst_lbls, exist_ok=True)


    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    for fn in tqdm(os.listdir(src_imgs), desc=f'CLAHE {split}'):
        if not fn.lower().endswith(('.jpg','.png')):
            continue

 
        img = cv2.imread(os.path.join(src_imgs, fn), cv2.IMREAD_GRAYSCALE)
    
        img_cl = clahe.apply(img)

        cv2.imwrite(os.path.join(dst_imgs, fn), img_cl)

        lbl = os.path.splitext(fn)[0] + '.txt'
        src_lbl = os.path.join(src_lbls, lbl)
        dst_lbl = os.path.join(dst_lbls, lbl)
        if os.path.exists(src_lbl):
            os.replace(src_lbl, dst_lbl) 