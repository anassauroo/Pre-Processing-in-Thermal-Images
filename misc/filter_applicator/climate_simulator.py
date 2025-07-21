import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

filename = list(uploaded.keys())[0]
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Simular clima úmido: desfoque
img_blur = cv2.GaussianBlur(img, (9, 9), 0)

# Simular clima nublado: ruído + escurecimento
noise = np.random.normal(loc=0, scale=25, size=img.shape).astype(np.uint8)
img_nublado = cv2.addWeighted(img, 0.6, noise, 0.4, -20)

# Aplicar filtro CLAHE sobre imagem borrada
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_blur)

fig, axs = plt.subplots(1, 4, figsize=(18, 5))

axs[0].imshow(img, cmap='gray')
axs[0].set_title('Imagem Original')
axs[0].axis('off')

axs[1].imshow(img_blur, cmap='gray')
axs[1].set_title('Simulação de Umidade')
axs[1].axis('off')

axs[2].imshow(img_nublado, cmap='gray')
axs[2].set_title('Simulação de Nebulosidade')
axs[2].axis('off')

axs[3].imshow(img_clahe, cmap='gray')
axs[3].set_title('CLAHE sobre imagem úmida')
axs[3].axis('off')

plt.tight_layout()
plt.savefig("comparacao_filtros.png")  
plt.show()
