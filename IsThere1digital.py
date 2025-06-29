#!/usr/bin/env python3
#python IsThere1digital.py input/bmp     --csv single.csv   --vis_out vis      --crop_out crops
"""
Batch fingerprint single‑presence detector (com visualização) + crop.
Novas funções:
* Salva bbox principal (digital central) no CSV.
* Salva recorte dessa bbox em `--crop_out`.
Mantido fluxo original; apenas adições.
python .\IsThere1digital.py input/bmp --csv single.csv --vis_out 
vis --crop_out crops
"""
import os
import csv
import argparse
import cv2
import numpy as np
from PIL import Image, PngImagePlugin
from filter import remove_lines_keep_fingerprints

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def write_metadata(path: str, single: bool) -> None:
    """Grava True/False no metadado da imagem (PNG ou JPEG)."""
    try:
        img = Image.open(path)
        tag = 'SingleFingerprint'
        val = 'True' if single else 'False'
        if img.format == 'PNG':
            meta = PngImagePlugin.PngInfo()
            for k, v in img.info.items():
                meta.add_text(k, str(v))
            meta.add_text(tag, val)
            img.save(path, pnginfo=meta)
        elif img.format in ('JPEG', 'JPG'):
            info = img.info or {}
            info[tag] = val
            img.save(path, 'JPEG', **info)
    except Exception:
        pass

# ------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------

def analyze_and_mark(mask: np.ndarray,
                     orig_bgr: np.ndarray,
                     min_area_ratio: float = 0.02,
                     solidity_thresh: float = 0.85,
                     overlap_thresh: float = 0.5):
    """Retorna (single, n_groups, solidity, vis, bbox_main, crop_img)."""
    h, w = mask.shape
    img_area = h * w

    # fecha + dilata
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    dilated = cv2.dilate(closed, k, iterations=1)

    # componentes conectados
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(dilated)
    big_idx = [i for i in range(1, n_lbl) if stats[i, cv2.CC_STAT_AREA] >= min_area_ratio * img_area]

    # caixas
    boxes = [[stats[i,0], stats[i,0]+stats[i,2], stats[i,1], stats[i,1]+stats[i,3]] for i in big_idx]

    # agrupar por sobreposição horizontal
    groups: list[list[list[int]]] = []
    for bx in boxes:
        for g in groups:
            gx1, gx2 = g[0][0], g[0][1]
            x1, x2   = bx[0], bx[1]
            inter = max(0, min(gx2, x2) - max(gx1, x1))
            min_w = min(gx2-gx1, x2-x1)
            if min_w and inter >= overlap_thresh * min_w:
                g.append(bx)
                break
        else:
            groups.append([bx])

    merged = []
    for g in groups:
        xs1 = [b[0] for b in g]; xs2 = [b[1] for b in g]
        ys1 = [b[2] for b in g]; ys2 = [b[3] for b in g]
        merged.append([min(xs1), max(xs2), min(ys1), max(ys2)])

    # bbox principal = maior área
    if merged:
        areas = [(b[1]-b[0])*(b[3]-b[2]) for b in merged]
        main_idx = int(np.argmax(areas))
        bbox_main = merged[main_idx]
        x1,x2,y1,y2 = bbox_main
        crop = orig_bgr[y1:y2, x1:x2].copy()
    else:
        bbox_main = [0,0,0,0]
        crop = None

    # visual
    vis = orig_bgr.copy()
    palette = [(0,255,0),(0,0,255),(255,0,0),(0,255,255),(255,255,0)]
    for gi,b in enumerate(merged):
        cv2.rectangle(vis,(b[0],b[2]),(b[1],b[3]),palette[gi%len(palette)],2)

    # solidity
    solidity=0.0
    cnts,_ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt=max(cnts,key=cv2.contourArea)
        area=cv2.contourArea(cnt)
        hull=cv2.convexHull(cnt)
        solidity = area / (cv2.contourArea(hull) or 1)
        cv2.drawContours(vis,[hull],-1,(255,255,0),1)

    single = (len(merged)==1 and solidity>=solidity_thresh)
    return single, len(merged), solidity, vis, bbox_main, crop

# ------------------------------------------------------------
# Directory loop + CSV
# ------------------------------------------------------------

def process_dir(inp: str,
                csv_path: str,
                vis_dir: str,
                crop_dir: str,
                min_area_ratio: float,
                solidity_thresh: float,
                overlap_thresh: float):
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)
    rows = []
    exts = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')

    for fname in sorted(os.listdir(inp)):
        if not fname.lower().endswith(exts):
            continue
        fpath = os.path.join(inp, fname)
        mask = remove_lines_keep_fingerprints(fpath)
        orig = cv2.imread(fpath)
        if mask is None or orig is None:
            print(f"[WARN] falha ao ler {fname}")
            continue

        single, grp, sol, vis, bbox, crop = analyze_and_mark(
            mask, orig,
            min_area_ratio=min_area_ratio,
            solidity_thresh=solidity_thresh,
            overlap_thresh=overlap_thresh
        )

        rows.append({
            'filename': fname,
            'single': single,
            #'groups': grp,
            #solidity': round(sol,3),
            'bbox_x1': bbox[0], 'bbox_x2': bbox[1],
            'bbox_y1': bbox[2], 'bbox_y2': bbox[3]
        })

        write_metadata(fpath, single)
        cv2.imwrite(os.path.join(vis_dir, fname), vis)
        if crop is not None and crop.size:
            crop_path = os.path.join(crop_dir, fname)

            # converte BGR → RGB e grava com 500 dpi
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            pil_crop.save(crop_path, dpi=(500, 500))

        print(f"{fname}: {'1 dedo' if single else 'multi'} (grp={grp}, sol={sol:.2f})")

    if rows:
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        with open(csv_path,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        print(f"CSV salvo em {csv_path}")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Detectar digital única + visual')
    ap.add_argument('input_dir')
    ap.add_argument('--csv', default='single.csv')
    ap.add_argument('--vis_out', default='vis')
    ap.add_argument('--crop_out', default='crops')
    ap.add_argument('--min_area_ratio', type=float, default=0.02)
    ap.add_argument('--solidity', type=float, default=0.85)
    ap.add_argument('--overlap_thresh', type=float, default=0.5)
    args=ap.parse_args()

    process_dir(
        inp=args.input_dir,
        csv_path=args.csv,
        vis_dir=args.vis_out,
        crop_dir=args.crop_out,
        min_area_ratio=args.min_area_ratio,
        solidity_thresh=args.solidity,
        overlap_thresh=args.overlap_thresh
    )

if __name__=='__main__':
    main()
