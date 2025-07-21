""" adiciona filtros que simulam umidade nas imagens para gerar dados de treinamento para o classificador de qualidade de imagem """

import cv2, numpy as np
from pathlib import Path
from tqdm import tqdm

def gamma_correction(img: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)],
        dtype=np.uint8)
    return cv2.LUT(img, table)

def gaussian_blur(img: np.ndarray, k: int = 5, sigma: int = 0) -> np.ndarray:
    return cv2.GaussianBlur(img, (k, k), sigma)

def simulate_haze(img: np.ndarray) -> np.ndarray:
    return gaussian_blur(gamma_correction(img, gamma=2.0), k=5)

IN_DIR  = Path('original_images')
OUT_DIR = Path('humidity_images')
OUT_DIR.mkdir(exist_ok=True)

IMG_EXTS = {'.jpg', '.jpeg'}

files = [p for p in IN_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
for path in tqdm(files, desc='processando'):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    hazy = simulate_haze(img)
    cv2.imwrite(str(OUT_DIR / path.name), hazy)

print(f'arquivos gerados em {OUT_DIR.resolve()}')