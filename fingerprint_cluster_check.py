#!/usr/bin/env python3
"""
python fingerprint_cluster_check.py Digitais_Descompactadas --jar sourceafis-3.18.1.jar
python fingerprint_cluster_check.py DB1_B --jar sourceafis-3.18.1.jar
Batch fingerprint multiple-presence detector usando SourceAFIS e DBSCAN.
Gera CSV com resultados e salva imagens com clusters se desejado.
"""
import os
import glob
import argparse
import cv2
import numpy as np
from PIL import Image
import jpype
import jpype.imports
import cbor2
import csv
from sklearn.cluster import DBSCAN

_vm_started = False

def start_jvm(jar_path):
    global _vm_started
    if not _vm_started:
        jars = glob.glob("*.jar")
        jpype.startJVM(classpath=jars)
        _vm_started = True

def extract_fingerprint_template(img, dpi, jar_path):
    start_jvm(jar_path)
    from com.machinezoo.sourceafis import FingerprintImage, FingerprintImageOptions, FingerprintTemplate
    pil_img = Image.fromarray(img)
    raw = pil_img.tobytes()
    opts = FingerprintImageOptions().dpi(dpi)
    fp_image = FingerprintImage(pil_img.width, pil_img.height, raw, opts)
    template = FingerprintTemplate(fp_image)
    return template.toByteArray()

def detect_multiple_fingerprints(template_cbor, eps=80, min_samples=5,
                                 cluster_size_ratio=0.1, min_centroid_dist=100):
    data = cbor2.loads(bytes(template_cbor))
    positionsX = np.array(data.get("positionsX", []))
    positionsY = np.array(data.get("positionsY", []))
    singularities = data.get("singularities", [])
    # Count deltas as secondary indicator
    deltas = [s for s in singularities if s.get('type') == 'delta']
    multiple_deltas = len(deltas) > 2
    # Prepare minutiae coords
    if positionsX.size == 0:
        return {"multiple_deltas": multiple_deltas,
                "num_clusters": 0,
                "multiple_clusters": False,
                "multiple_fingerprints_detected": multiple_deltas,
                "positions": np.empty((0,2)),
                "labels": []}
    coords = np.column_stack((positionsX, positionsY))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    # Compute cluster sizes
    unique_labels = [l for l in set(labels) if l != -1]
    counts = {l: np.sum(labels == l) for l in unique_labels}
    # Filter clusters by size ratio
    total = len(labels)
    large = [l for l,c in counts.items() if c >= cluster_size_ratio * total]
    # Compute centroids for large clusters
    centroids = []
    for l in large:
        pts = coords[labels == l]
        centroids.append(np.mean(pts, axis=0))
    centroids = np.array(centroids)
    # Merge centroids closer than min_centroid_dist
    visited = set()
    groups = []
    for i in range(len(centroids)):
        if i in visited: continue
        stack = [i]
        comp = []
        while stack:
            j = stack.pop()
            if j in visited: continue
            visited.add(j)
            comp.append(j)
            # find neighbors
            dists = np.linalg.norm(centroids - centroids[j], axis=1)
            for k, d in enumerate(dists):
                if d < min_centroid_dist and k not in visited:
                    stack.append(k)
        groups.append(comp)
    num_clusters = len(groups)
    multiple_clusters = num_clusters > 1
    result = multiple_clusters or multiple_deltas
    return {
        "multiple_deltas": multiple_deltas,
        "num_clusters": num_clusters,
        "multiple_clusters": multiple_clusters,
        "multiple_fingerprints_detected": result,
        "positions": coords,
        "labels": labels
    }

def draw_clusters(image, coords, labels, out_path):
    colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,255,0)]
    img = image.copy()
    for idx, (x,y) in enumerate(coords):
        l = labels[idx]
        color = (150,150,150) if l == -1 else colors[l % len(colors)]
        cv2.circle(img, (int(x),int(y)), 2, color, -1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)

def process_directory(input_dir, jar_path, dpi, eps, min_samples,
                      cluster_size_ratio, min_centroid_dist,
                      csv_path, img_out_dir):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    results = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(exts): continue
        path = os.path.join(input_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[ERRO] Falha ao carregar {fname}")
            continue
        try:
            tpl = extract_fingerprint_template(img, dpi, jar_path)
            res = detect_multiple_fingerprints(
                tpl, eps, min_samples,
                cluster_size_ratio, min_centroid_dist
            )
            print(f"\nArquivo: {fname}")
            print(f"  Múltiplos deltas: {res['multiple_deltas']}")
            print(f"  Clusters de minutiae filtrados: {res['num_clusters']}")
            print(f"  Clusters múltiplos: {res['multiple_clusters']}")
            print(f"  Resultado final: {'MÚLTIPLAS DIGITAIS' if res['multiple_fingerprints_detected'] else '1 digital'}")
            results.append({
                'filename': fname,
                'multiple_deltas': res['multiple_deltas'],
                'num_clusters': res['num_clusters'],
                'multiple_clusters': res['multiple_clusters'],
                'multiple_fingerprints_detected': res['multiple_fingerprints_detected']
            })
            if img_out_dir and res['positions'].shape[0] > 0:
                cls_img_path = os.path.join(img_out_dir, fname)
                orig = cv2.imread(path)
                draw_clusters(orig, res['positions'], res['labels'], cls_img_path)
        except Exception as e:
            print(f"[ERRO] {fname}: {e}")
    # CSV output
    if csv_path and results:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV salvo em: {csv_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Detectar múltiplas digitais em diretório"
    )
    parser.add_argument('input_dir', help='Diretório com imagens')
    parser.add_argument('--jar', required=True, help='sourceafis-<versão>.jar')
    parser.add_argument('--dpi', type=int, default=500)
    parser.add_argument('--eps', type=float, default=50)
    parser.add_argument('--min_samples', type=int, default=5)
    parser.add_argument('--cluster_size_ratio', type=float, default=0.1,
                        help='Razão mínima de tamanho de cluster')
    parser.add_argument('--min_centroid_dist', type=float, default=100,
                        help='Distância mínima (px) para separar clusters')
    parser.add_argument('--csv', default='resultados.csv', help='Arquivo CSV de saída')
    parser.add_argument('--img_out', default='clusters_img',
                        help='Diretório para imagens dos clusters')
    args = parser.parse_args()
    process_directory(
        args.input_dir, args.jar, args.dpi,
        args.eps, args.min_samples,
        args.cluster_size_ratio, args.min_centroid_dist,
        args.csv, args.img_out
    )

if __name__ == '__main__':
    main()
