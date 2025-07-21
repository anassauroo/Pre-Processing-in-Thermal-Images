import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Simular clima úmido com blur
img_umida = cv2.GaussianBlur(img, (9, 9), 0)

# CLAHE 
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_umida)

# detailEnhance
img_bgr = cv2.cvtColor(img_umida, cv2.COLOR_GRAY2BGR)
img_detail = cv2.detailEnhance(img_bgr, sigma_s=10, sigma_r=0.15)
img_detail_gray = cv2.cvtColor(img_detail, cv2.COLOR_BGR2GRAY)

# Filtro de nitidez (sharpening)
kernel_sharpen = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
img_sharpen = cv2.filter2D(img_umida, -1, kernel_sharpen)

# Guided filter 
try:
    import cv2.ximgproc as xip
except:
    !pip install opencv-contrib-python
    import cv2.ximgproc as xip

img_guided = xip.guidedFilter(guide=img, src=img_umida, radius=5, eps=20, dDepth=-1)
titles = ['Original', 'CLAHE', 'DetailEnhance', 'Sharpen', 'Guided Filter']
images = [img_umida, img_clahe, img_detail_gray, img_sharpen, img_guided]
plt.figure(figsize=(18,4))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()