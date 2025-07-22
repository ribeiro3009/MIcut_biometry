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

    # Detecta linhas
    vertical_lines   = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    all_lines        = cv2.bitwise_or(vertical_lines, horizontal_lines)

    # Remove linhas e limpa ruídos
    fingerprints_only = cv2.bitwise_and(thresh, cv2.bitwise_not(all_lines))
    kernel_small      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned           = cv2.morphologyEx(fingerprints_only, cv2.MORPH_OPEN, kernel_small)

    # Dilata para realçar cristas
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filtered      = cv2.dilate(cleaned, kernel_dilate, iterations=1)

    return filtered


def segment_fingerprint(
    image_path: str,
    min_area_ratio: float    = 0.02,
    solidity_thresh: float   = 0.85,
    overlap_thresh: float    = 0.5,
):
    """
    Detecta se há uma única impressão digital.
    Retorna sempre um dict com:
      - is_single: bool
      - box: [x1, x2, y1, y2] ou None
      - cropped_image: np.ndarray ou None
      - mask: np.ndarray[bool] ou None (máscara do ROI recortado)
    """
    try:
        # 1) Segmentação inicial: linhas removidas, só cristas
        mask = remove_lines_keep_fingerprints(image_path)
        orig_bgr = cv2.imread(image_path)
        if mask is None or orig_bgr is None:
            return None

        h, w     = mask.shape
        img_area = h * w

        # 2) Limpeza morfológica e rotulagem de componentes
        k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        dilated = cv2.dilate(closed, k, iterations=1)

        n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(dilated)
        # Filtra componentes grandes
        big_idx = [
            i for i in range(1, n_lbl)
            if stats[i, cv2.CC_STAT_AREA] >= min_area_ratio * img_area
        ]
        # Constrói boxes [x1,x2,y1,y2]
        boxes = [
            [stats[i,0], stats[i,0]+stats[i,2],
             stats[i,1], stats[i,1]+stats[i,3]]
            for i in big_idx
        ]

        # 3) Agrupamento de boxes por overlap horizontal
        groups = []
        for bx in boxes:
            for g in groups:
                gx1, gx2 = g[0][0], g[0][1]
                x1,  x2  = bx[0], bx[1]
                inter   = max(0, min(gx2, x2) - max(gx1, x1))
                min_w   = min(gx2-gx1, x2-x1)
                if min_w and inter >= overlap_thresh * min_w:
                    g.append(bx)
                    break
            else:
                groups.append([bx])

        # 4) Mescla grupos em uma única bbox cada
        merged = [
            [
                min(b[0] for b in g), max(b[1] for b in g),
                min(b[2] for b in g), max(b[3] for b in g)
            ]
            for g in groups
        ]

        # 5) Seleciona maior bbox e recorta
        if merged:
            areas    = [(b[1]-b[0])*(b[3]-b[2]) for b in merged]
            main_idx = int(np.argmax(areas))
            x1, x2, y1, y2 = merged[main_idx]
            crop = (
                orig_bgr[y1:y2, x1:x2].copy()
                if (y2>y1 and x2>x1) else None
            )
        else:
            x1 = x2 = y1 = y2 = None
            crop = None

        # 6) Condição de “single print” e solidity
        cnts, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if cnts:
            cnt  = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            solidity = area / (cv2.contourArea(hull) or 1)
        else:
            solidity = 0.0

        is_single = (len(merged) == 1)

        # 7) Recorta também a máscara dilatada para ROI_mask
        mask_crop = None
        if crop is not None:
            mask_crop = dilated[y1:y2, x1:x2].astype(bool)

        return {
            "is_single":      is_single,
            "box":            [x1, x2, y1, y2] if crop is not None else None,
            "cropped_image":  crop,
            "mask":           mask_crop
        }

    except Exception as e:
        print(f"[ERROR] {image_path}: {e}")
        return None
