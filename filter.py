#!/usr/bin/env python3
"""
Aplica o filtro remove_lines_keep_fingerprints a todas as imagens de um diretório.
"""

import os
import cv2
import numpy as np

def remove_lines_keep_fingerprints(image_path):
    """Remove linhas verticais/horizontais mantendo as cristas das digitais."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarização adaptativa
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 9
    )

    # Kernels para linhas
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))

    # Detecta e combina linhas
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    all_lines = cv2.bitwise_or(vertical_lines, horizontal_lines)

    # Remove linhas da máscara e limpa ruídos
    fingerprints_only = cv2.bitwise_and(thresh, cv2.bitwise_not(all_lines))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(fingerprints_only, cv2.MORPH_OPEN, kernel_small)

    # Dilata para realçar cristas
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filtered = cv2.dilate(cleaned, kernel_dilate, iterations=1)
    return filtered

def main():
    input_dir = "bmp"
    output_dir = "filtered"
    os.makedirs(output_dir, exist_ok=True)

    # Extensões de imagem válidas
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(exts):
            continue
        in_path = os.path.join(input_dir, fname)
        out_name = os.path.splitext(fname)[0] + "_filtered.jpg"
        out_path = os.path.join(output_dir, out_name)

        # Aplica filtro
        result = remove_lines_keep_fingerprints(in_path)
        if result is not None:
            cv2.imwrite(out_path, result)
            print(f"Salvo: {out_path}")

if __name__ == "__main__":
    main()
