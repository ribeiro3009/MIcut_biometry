import os
import glob
import cv2
import numpy as np
from PIL import Image
import jpype
import jpype.imports
import cbor2
from sklearn.cluster import DBSCAN

# --- Gerenciamento da JVM ---
_vm_started = False

def start_jvm(jars_to_load: list[str]):
    """Inicia a JVM do JPype se ainda não estiver ativa, usando os JARs fornecidos."""
    global _vm_started
    if not _vm_started:
        if not jars_to_load:
            raise RuntimeError("Nenhum arquivo JAR fornecido para iniciar a JVM.")
        jvm_path ="C:\\Users\\lflma\\AppData\\Local\\JetBrains\\Installations\\ReSharperPlatformVs17_224b7884\\Bin.ExtSvc\\jbr\\bin\\server\\jvm.dll"
        jpype.startJVM(jvm_path, classpath=jars_to_load)
        _vm_started = True

# --- Lógica Central (adaptada de fingerprint_cluster_check.py) ---

def extract_fingerprint_template(img, dpi=500):
    """Extrai o template da digital usando SourceAFIS."""

    from com.machinezoo.sourceafis import FingerprintImage, FingerprintImageOptions, FingerprintTemplate

    # Garante que a imagem está em escala de cinza
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pil_img = Image.fromarray(img)
    raw_data = pil_img.tobytes()
    
    opts = FingerprintImageOptions().dpi(dpi)
    fp_image = FingerprintImage(pil_img.width, pil_img.height, raw_data, opts)
    template = FingerprintTemplate(fp_image)
    return template.toByteArray()

def detect_clusters_and_singularities(template_cbor, eps=50, min_samples=5, cluster_size_ratio=0.1, min_centroid_dist=100):
    """Analisa o template para contar minúcias, singularidades e clusters."""
    data = cbor2.loads(bytes(template_cbor))
    
    positionsX = np.array(data.get("positionsX", []))
    positionsY = np.array(data.get("positionsY", []))
    singularities = data.get("singularities", [])
    
    minutiae_count = len(positionsX)
    singularities_count = len(singularities)
    
    if minutiae_count < min_samples:
        return {
            "minutiae_count": minutiae_count,
            "singularities_count": singularities_count,
            "cluster_count": 1 if minutiae_count > 0 else 0
        }

    coords = np.column_stack((positionsX, positionsY))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    
    unique_labels = [l for l in set(labels) if l != -1]
    if not unique_labels:
        return {
            "minutiae_count": minutiae_count,
            "singularities_count": singularities_count,
            "cluster_count": 0
        }

    # Lógica de filtragem e merge de clusters do script original
    counts = {l: np.sum(labels == l) for l in unique_labels}
    total_minutiae_in_clusters = len(labels[labels != -1])
    if total_minutiae_in_clusters == 0:
        return {
            "minutiae_count": minutiae_count,
            "singularities_count": singularities_count,
            "cluster_count": 0
        }

    large_clusters = [l for l, c in counts.items() if c >= cluster_size_ratio * total_minutiae_in_clusters]
    
    if not large_clusters:
        return {
            "minutiae_count": minutiae_count,
            "singularities_count": singularities_count,
            "cluster_count": 0
        }

    centroids = np.array([coords[labels == l].mean(axis=0) for l in large_clusters])
    
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
            dists = np.linalg.norm(centroids - centroids[j], axis=1)
            for k, d in enumerate(dists):
                if d < min_centroid_dist and k not in visited:
                    stack.append(k)
        groups.append(comp)
        
    num_clusters = len(groups)

    return {
        "minutiae_count": minutiae_count,
        "singularities_count": singularities_count,
        "cluster_count": num_clusters
    }

# --- Função de Integração com o Pipeline ---

def analyze_minutiae_from_image(img_array: np.ndarray):
    """
    Função principal chamada pelo pipeline.
    Analisa um array de imagem usando SourceAFIS via jpype.
    """
    default_return = {"minutiae_count": 0, "singularities_count": 0, "cluster_count": 0}
    
    try:
        if img_array is None:
            return default_return

        template_cbor = extract_fingerprint_template(img_array)
        analysis_results = detect_clusters_and_singularities(template_cbor)
        
        return analysis_results

    except Exception as e:
        # Captura exceções do jpype/java
        # print(f"Erro na análise de minúcias para {os.path.basename(cropped_image_path)}: {e}")
        return default_return
