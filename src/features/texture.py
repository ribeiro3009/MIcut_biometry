
# features/texture.py

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def morphological_skeleton(bin_img: np.ndarray) -> np.ndarray:
    """
    Realiza o esqueleto morfológico (thinning) de uma imagem binária usando
    operações de erosão e abertura até convergência.
    """
    skeleton = np.zeros_like(bin_img)
    # elemento de cruz 3x3
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = bin_img.copy()
    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def block_orientation_std(gray: np.ndarray, roi_mask: np.ndarray, block: int = 16) -> float:
    """
    Calcula o desvio-padrão circular de orientações em blocos dentro da ROI.
    """
    angles = []
    H, W = gray.shape
    for y in range(0, H, block):
        for x in range(0, W, block):
            patch = gray[y:y+block, x:x+block]
            m = roi_mask[y:y+block, x:x+block]
            # exige que 30% do bloco pertença à ROI
            if m.sum() < (block * block * 0.3):
                continue
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            v_x = np.sum(2 * gx * gy)
            v_y = np.sum(gx * gx - gy * gy)
            theta = 0.5 * np.arctan2(v_x, v_y)
            angles.append(theta)
    if not angles:
        return 0.0
    theta2 = 2 * np.array(angles)
    C = np.mean(np.cos(theta2))
    S = np.mean(np.sin(theta2))
    R = np.hypot(C, S)
    circ_std = np.sqrt(-2.0 * np.log(R)) if R > 0 else np.pi / np.sqrt(3)
    return float(circ_std * (180.0 / np.pi))


def analyze_texture(
    cropped_image: np.ndarray,
    roi_mask: np.ndarray = None
) -> dict:
    """
    Calcula métricas de textura:
      • sharpness (variância do Laplacian),
      • desvio padrão circular de orientação (via esqueleto morfológico),
      • contraste local (RMS).

    Args:
        cropped_image (np.ndarray): grayscale uint8 recortada.
        roi_mask       (np.ndarray): máscara boolean da ROI.

    Returns:
        dict: {'sharpness', 'orientation_std', 'contrast'}.
    """
    default_return = {"sharpness": 0.0, "orientation_std": 0.0, "contrast": 0.0}
    if cropped_image is None:
        return default_return

    try:
        gray = cropped_image

        # 1) Sharpness: variância do Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(lap.var())

        # 2) ROI fallback Otsu se não fornecida
        if roi_mask is None:
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, tmp = cv2.threshold(
                blur, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            kernel = np.ones((5, 5), np.uint8)
            tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel)
            tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN,  kernel)
            roi_mask = tmp.astype(bool)

        # 3) Usa o método de blocos para orientação
        orientation_std = block_orientation_std(gray, roi_mask)

        # 4) Contrast (RMS)
        _, stddev = cv2.meanStdDev(gray)
        contrast = float(stddev[0, 0])

        return {
            "sharpness": sharpness,
            "orientation_std": orientation_std,
            "contrast": contrast
        }

    except Exception:
        logger.exception("analyze_texture failed")
        return default_return