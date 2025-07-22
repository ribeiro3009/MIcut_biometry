import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def analyze_shape(mask):
    """
    Calculates shape-based metrics (solidity and coverage) from a binary mask.

    Args:
        mask (numpy.ndarray): The binary mask of the fingerprint.

    Returns:
        dict: A dictionary with 'solidity' and 'coverage'.
              Returns default values if mask is invalid or has no contour.
    """
    default_return = {"solidity": 0.0, "coverage": 0.0}

    # 1) Validação de entrada e conversão única de tipo
    if mask is None:
        return default_return
    if mask.dtype != np.uint8:
        # Garante uint8 binário (0 ou 1) só uma vez
        mask = (mask > 0).astype(np.uint8)
    if not mask.any():
        return default_return

    try:
        # 2) Crop para ROI: reduz tamanho antes de chamar findContours
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        submask = mask[y0:y1+1, x0:x1+1]

        # 3) Detecção de contornos só no recorte
        contours, _ = cv2.findContours(submask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return default_return

        # 4) Seleciona o maior contorno
        largest = max(contours, key=cv2.contourArea)

        # 5) Aproxima o contorno (menos pontos → hull mais rápido)
        eps = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, eps, True)

        # 6) Cálculo de solidity usando o contorno simplificado
        area = cv2.contourArea(approx)
        hull = cv2.convexHull(approx)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        # 7) Cálculo de coverage sobre bounding-box do contorno simplificado
        _, _, w, h = cv2.boundingRect(approx)
        bbox_area = w * h
        coverage = float(area) / bbox_area if bbox_area > 0 else 0.0

        return {"solidity": solidity, "coverage": coverage}

    except Exception:
        # 8) Logging em vez de print, para não poluir stdout
        logger.exception("analyze_shape failed")
        return default_return
