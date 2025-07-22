import cv2
import numpy as np

def get_lines_mask(gray_image: np.ndarray) -> np.ndarray:
    """
    Gera uma máscara de linhas a partir de uma imagem em tons de cinza (preferencialmente já recortada).
    """
    if gray_image is None:
        return np.zeros((1, 1), dtype=bool)

    # Inverte a imagem e usa um threshold simples para destacar as linhas escuras.
    inverted_gray = cv2.bitwise_not(gray_image)
    _, line_thresh = cv2.threshold(inverted_gray, 100, 255, cv2.THRESH_BINARY)
    
    # Usa kernels longos para encontrar seletivamente as linhas retas.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    
    vertical_lines = cv2.morphologyEx(line_thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    horizontal_lines = cv2.morphologyEx(line_thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    all_lines = cv2.bitwise_or(vertical_lines, horizontal_lines)
    
    # Dilata a máscara final para garantir que cubra a espessura total das linhas.
    dilated_lines = cv2.dilate(all_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    
    return dilated_lines.astype(bool)
