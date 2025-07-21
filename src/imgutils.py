import cv2, numpy as np, pathlib

FIX_SIZE = (512, 512)
EXTS = {'.jpg', '.jpeg'}

def load_gray(path: pathlib.Path) -> np.ndarray:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, FIX_SIZE, interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32)
    g /= g.max() + 1e-9
    return g

# ---------- metricas -----------------------------------------
def entropy(img: np.ndarray) -> float:
    h = cv2.calcHist([img], [0], None, [256], [0, 1]).ravel()
    h /= h.sum() + 1e-7
    return -np.sum(h * np.log2(h + 1e-7))

def lap_var(img: np.ndarray) -> float:
    return cv2.Laplacian(img, cv2.CV_32F).var()

# ---------- filtros --------------------------------------------
def clahe_guided(img_u8: np.ndarray, radius=5, eps=1e-1) -> np.ndarray:
    cla = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img_u8)
    g   = cv2.ximgproc.guidedFilter(guide=cla, src=cla,
                                    radius=radius, eps=eps)
    return np.clip(g, 0, 255).astype(np.uint8)

def bilateral(img_u8: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img_u8, d=9, sigmaColor=75, sigmaSpace=75)
