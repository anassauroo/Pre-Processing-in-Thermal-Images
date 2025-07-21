import cv2
import glob
import os

input_dir  = '/content/original_images'   # pasta de entrada
output_dir = '/content/CLAHE_images'      # pasta de saída

clahe = cv2.createCLAHE(
    clipLimit   = 2.0,    
    tileGridSize = (8, 8)
)
for filepath in glob.glob(os.path.join(input_dir, '*')):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    
    img_clahe = clahe.apply(img)
    
    filename  = os.path.basename(filepath)
    save_path = os.path.join(output_dir, filename)
    
    cv2.imwrite(save_path, img_clahe)