#!/usr/bin/env python3
"""
Batch fingerprint filter and mark tool.
Aplica remoção de linhas horizontais/verticais e marca contorno da impressão para todas as imagens em um diretório.
"""
import os
import argparse
import cv2
import numpy as np


def remove_lines_keep_fingerprints(image_path):
    """Remove linhas verticais/horizontais mantendo apenas as cristas das digitais."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 5
    )
    # kernels para linhas
    vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vert = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_k)
    horz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horz_k)
    lines = cv2.bitwise_or(vert, horz)
    only = cv2.bitwise_and(thresh, cv2.bitwise_not(lines))
    # limpeza e dilatação
    small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(only, cv2.MORPH_OPEN, small_k)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filtered = cv2.dilate(cleaned, dilate_k, iterations=1)
    return filtered


def mark_fingerprint(orig_img, mask, output_path):
    """Desenha o contorno principal da impressão e salva a imagem marcada."""
    if mask is None:
        print(f"Máscara inválida para {output_path}")
        return
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print(f"Nenhum contorno encontrado em {output_path}")
        return
    cnt = max(contours, key=cv2.contourArea)
    marked = orig_img.copy()
    cv2.drawContours(marked, [cnt], -1, (0, 255, 0), 2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, marked)
    print(f"Salvo: {output_path}")


def process_directory(input_dir, out_dir):
    """Processa todas as imagens do diretório recursivamente."""
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(exts):
                continue
            input_path = os.path.join(root, fname)
            orig = cv2.imread(input_path)
            filtered = remove_lines_keep_fingerprints(input_path)
            mask = (filtered > 0).astype(np.uint8) if filtered is not None else None
            rel = os.path.relpath(root, input_dir)
            target_dir = os.path.join(out_dir, rel)
            name, ext = os.path.splitext(fname)
            output_path = os.path.join(target_dir, f"{name}_marked{ext}")
            mark_fingerprint(orig, mask, output_path)


def main():
    parser = argparse.ArgumentParser(description="Batch fingerprint filter and mark tool")
    parser.add_argument('input_dir', help='Diretório contendo imagens de digitais')
    parser.add_argument('--out-dir', default='marked_fingerprints', help='Diretório de saída')
    args = parser.parse_args()
    if not os.path.isdir(args.input_dir):
        print(f"Erro: {args.input_dir} não é um diretório válido.")
        return
    process_directory(args.input_dir, args.out_dir)

if __name__ == '__main__':
    main()
