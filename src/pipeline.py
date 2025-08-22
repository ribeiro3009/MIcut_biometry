import os
import glob
import sys
import subprocess
import multiprocessing
import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
from functools import partial
import cbor2
import jpype, jpype.imports

# Import feature analysis functions
from features.ml_segmentation import create_columns_from_cuts, segment_columns_with_ml
#from features.minutiae import analyze_minutiae_from_image, start_jvm
from features.minutiae import start_jvm, extract_fingerprint_template, detect_clusters_and_singularities

from features.shape import analyze_shape
from features.texture import analyze_texture
from features.frequency import analyze_ridge_frequency

from features.dfiqi import dfiqi_on_image, DFIQIParams

# --- Configuration ---
# Build paths relative to the project root to make the script runnable from anywhere
def get_project_root():                                                                              
  """ Retorna o caminho raiz do projeto, seja rodando como script ou como executável. """              
  if getattr(sys, 'frozen', False):
        return sys._MEIPASS
  else:                                                                                                 
        # Se estiver rodando como um script normal                                                       
        # A lógica original está correta.                                                                 
        script_dir = os.path.dirname(os.path.abspath(__file__))                          
        return os.path.dirname(script_dir)                          
        
PROJECT_ROOT = get_project_root()  
INPUT_DIR_CUTS = os.path.join(PROJECT_ROOT, "data", "input", "Fingerprints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")
COLUMN_DIR = os.path.join(OUTPUT_DIR, "merged_columns_from_pipeline")
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
MODEL_PATH = os.path.join(PROJECT_ROOT, "bin", "best_detector_model_v2.pth")
NFIQ2_EXECUTABLE_PATH = os.path.join(PROJECT_ROOT, "bin", "NFIQ2", "bin", "NFIQ2.exe")
FINAL_RESULTS_CSV = os.path.join(OUTPUT_DIR, "full_analysis.csv")

# --- Utility Functions ---
def setup_directories():
    os.makedirs(CROPS_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(COLUMN_DIR, exist_ok=True)

# --- Pipeline Stages ---
def run_stage_1_ml_segmentation(input_dir, column_dir, model_path, crops_dir, masks_dir):
    print("--- Stage 1: Creating Columns and Segmenting with ML Model ---")
    
    # Part 1: Create column images from individual cuts
    print(f"Creating column images from cuts in '{input_dir}'...")
    column_paths = create_columns_from_cuts(input_dir, column_dir)
    if not column_paths:
        print("No columns were created. Aborting Stage 1.")
        return pl.DataFrame()
    print(f"Successfully created {len(column_paths)} column images.")

    # Part 2: Run ML model prediction on the columns
    return segment_columns_with_ml(column_paths, model_path, crops_dir, masks_dir)

def run_stage_2_nfiq2(crop_files: list[str]):
    print("\n--- Stage 2: Running NFIQ2 Analysis ---")
    
    root_dir = os.getcwd()
    batch_file_path = os.path.join(root_dir, "nfiq2_batch.txt")
    temp_output_path = os.path.join(root_dir, "nfiq2_output.csv")
    
    if not crop_files:
        print("No cropped images found to analyze with NFIQ2.")
        return pl.DataFrame()

    with open(batch_file_path, 'w') as f:
        for path in crop_files:
            f.write(f"{os.path.abspath(path)}\n")

    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    command = [
        NFIQ2_EXECUTABLE_PATH, "-f", batch_file_path, "-o", temp_output_path,
        "-j", str(min(os.cpu_count() * 2, 8)), "-F"
    ]
    
    print(f"Executing NFIQ2...")
    try:
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        print("NFIQ2 analysis complete.")
    except subprocess.CalledProcessError as e:
        print(f"NFIQ2 execution failed: {e.stderr.decode(errors='ignore') if e.stderr else 'Unknown error'}")
        if os.path.exists(batch_file_path): os.remove(batch_file_path)
        return pl.DataFrame()

    if os.path.exists(batch_file_path): os.remove(batch_file_path)

    if not os.path.exists(temp_output_path):
        print("NFIQ2 did not produce an output file.")
        return pl.DataFrame()

    try:
        nfiq_df = pl.read_csv(
            temp_output_path,
            encoding='latin-1',
            null_values=["NA"]
        ).rename({"Filename": "filename", "QualityScore": "NFIQ2"})
        nfiq_df = nfiq_df.select(["filename", "NFIQ2"])
        nfiq_df = nfiq_df.with_columns(
            pl.col("filename").map_elements(lambda x: os.path.basename(x.strip('"')), return_dtype=pl.Utf8)
        )
    finally:
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
            
    return nfiq_df

def _imread_unicode(path, flags):
    """cv2.imread wrapper for unicode paths."""
    try:
        with open(path, 'rb') as f:
            img_np = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(img_np, flags)
        return img
    except (IOError, FileNotFoundError):
        # This mimics cv2.imread behavior of returning None on failure
        return None


# esse trecho do codigo tbm calcula as minucias, so que ele pega outros parametros do que minutiae.py, inclusive percebi diferença de resultados do DFIQI quando utiliza essa função e a reutilização da função do minutiae.py
# outro problema de usar esse trecho do codigo , é ter q compilar duas vezes o SOURCEAFIS
'''
def _extract_minutiae_points_sourceafis(img_gray: np.ndarray, dpi: int = 500) -> list[tuple[float, float]]:
    """
    Gera o template (CBOR) via SourceAFIS e extrai coordenadas (x,y) das minúcias.
    Usa variantes comuns do CBOR: positionsX/positionsY ou lista 'minutiae'.
    """
    # importa classes Java *após* a JVM estar iniciada no worker
    from com.machinezoo.sourceafis import FingerprintImage, FingerprintImageOptions, FingerprintTemplate

    if img_gray.ndim == 3 and img_gray.shape[2] == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    pil = Image.fromarray(img_gray)
    opts = FingerprintImageOptions().dpi(dpi)
    fp_image = FingerprintImage(pil.width, pil.height, pil.tobytes(), opts)
    template = FingerprintTemplate(fp_image).toByteArray()

    data = cbor2.loads(bytes(template))
    pts: list[tuple[float, float]] = []

    if "positionsX" in data and "positionsY" in data:
        xs = data.get("positionsX", [])
        ys = data.get("positionsY", [])
        pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
    elif "minutiae" in data and isinstance(data["minutiae"], list):
        for mi in data["minutiae"]:
            if not isinstance(mi, dict):
                continue
            x = mi.get("x", mi.get("positionX", mi.get("position", {}).get("x", 0.0)))
            y = mi.get("y", mi.get("positionY", mi.get("position", {}).get("y", 0.0)))
            pts.append((float(x), float(y)))

    return pts

def analyze_python_features(crop_path):
    try:
        filename = os.path.basename(crop_path)
        
        mask_path = os.path.join(MASKS_DIR, filename.replace(".bmp", ".png"))
        roi_mask = _imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if roi_mask is not None:
            roi_mask = (roi_mask > 0).astype(np.uint8)

        cropped_image = _imread_unicode(crop_path, cv2.IMREAD_GRAYSCALE)
        if cropped_image is None:
            return {"filename": filename}

        # --- Features já existentes ---
        minutiae_data = analyze_minutiae_from_image(cropped_image)
        shape_data    = analyze_shape(roi_mask)
        texture_data  = analyze_texture(cropped_image, roi_mask)
        frequency_data= analyze_ridge_frequency(cropped_image, roi_mask)

        # --- DFIQI: extrai minúcias (x,y) e calcula GQS/LQSsum ---
        dfiqi_fields = {}
        try:
            features_xy = _extract_minutiae_points_sourceafis(cropped_image, dpi=500)
            if features_xy:  # só roda DFIQI se houver pelo menos 1 feature
                dfiqi_out = dfiqi_on_image(cropped_image, features_xy, DFIQIParams(dpi=500.0))
                g = dfiqi_out["global"]
                # Campos compactos para o DataFrame
                dfiqi_fields = {
                    "DFIQI_nFEAT":         g["nFEAT"],
                    "DFIQI_LQSsum":        g["LQSsum"],
                    "DFIQI_ValueGQS":      g["ValueGQS"],
                    "DFIQI_ComplexityGQS": g["ComplexityGQS"],
                    "DFIQI_DifficultyGQS": g["DifficultyGQS"],
                }
                # (Se quiser probabilidades detalhadas, você pode adicionar mais colunas aqui)
        except Exception as e:
            # Não derruba a pipeline se DFIQI falhar para um item
            print(f"DFIQI failed for {filename}: {e}")

        return {
            "filename": filename,
            **minutiae_data,
            **shape_data,
            **texture_data,
            **frequency_data,
            **dfiqi_fields,     # <--- NOVO: colunas DFIQI
        }
    except Exception as e:
        print(f"Error in Python analysis for {os.path.basename(crop_path)}: {e}")
        return {"filename": os.path.basename(crop_path), "error": str(e)}

'''

def analyze_python_features(crop_path):
    try:
        filename = os.path.basename(crop_path)

        # ROI mask
        mask_path = os.path.join(MASKS_DIR, filename.replace(".bmp", ".png"))
        roi_mask = _imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if roi_mask is not None:
            roi_mask = (roi_mask > 0).astype(np.uint8)

        # Imagem recortada
        cropped_image = _imread_unicode(crop_path, cv2.IMREAD_GRAYSCALE)
        if cropped_image is None:
            return {"filename": filename}

        # ------------------------------
        # (A) Extrai o TEMPLATE uma única vez (SourceAFIS)
        # ------------------------------
        template_cbor = extract_fingerprint_template(cropped_image, dpi=500)

        # (A1) Contagens/cluster/singularidades reaproveitando minutiae.py
        minutiae_data = detect_clusters_and_singularities(template_cbor)

        # (A2) Extrai (x,y) das minúcias a partir do CBOR (sem reexecutar SourceAFIS)
        features_xy = []
        try:
            data = cbor2.loads(bytes(template_cbor))
            xs = data.get("positionsX", [])
            ys = data.get("positionsY", [])
            if xs and ys:
                features_xy = [(float(x), float(y)) for x, y in zip(xs, ys)]
            elif isinstance(data.get("minutiae", None), list):
                for mi in data["minutiae"]:
                    if isinstance(mi, dict):
                        x = mi.get("x", mi.get("positionX", mi.get("position", {}).get("x", 0.0)))
                        y = mi.get("y", mi.get("positionY", mi.get("position", {}).get("y", 0.0)))
                        features_xy.append((float(x), float(y)))
        except Exception as e:
            print(f"CBOR parse failed for {filename}: {e}")
            features_xy = []

        # ------------------------------
        # (B) Demais features
        # ------------------------------
        shape_data     = analyze_shape(roi_mask)
        texture_data   = analyze_texture(cropped_image, roi_mask)
        frequency_data = analyze_ridge_frequency(cropped_image, roi_mask)

        # ------------------------------
        # (C) DFIQI usando as minúcias extraídas acima
        
        # se quiser retirar a parte do DFIQI
        #'''

        dfiqi_fields = {}
        try:
            if features_xy:
                dfiqi_out = dfiqi_on_image(cropped_image, features_xy, DFIQIParams(dpi=500.0))
                g = dfiqi_out["global"]
                dfiqi_fields = {
                    # "DFIQI_nFEAT": g["nFEAT"],            
                    "DFIQI_LQSsum":        g["LQSsum"],
                    "DFIQI_ValueGQS":      g["ValueGQS"],
                    "DFIQI_ComplexityGQS": g["ComplexityGQS"],
                    "DFIQI_DifficultyGQS": g["DifficultyGQS"],
                }
        except Exception as e:
            print(f"DFIQI failed for {filename}: {e}")

        return {
            "filename": filename,
            **minutiae_data,
            **shape_data,
            **texture_data,
            **frequency_data,
            **dfiqi_fields,
        }
    #'''

    except Exception as e:
        print(f"Error in Python analysis for {os.path.basename(crop_path)}: {e}")
        return {"filename": os.path.basename(crop_path), "error": str(e)}



def run_stage_3_python_analysis(crop_paths: list[str], jvm_jars: list[str]):
    print("--- Stage 3: Running Python-based Feature Analysis (Parallel) ---")
    if not crop_paths:
        print("No cropped images found for Python analysis.")
        return pl.DataFrame()
    results = []
    # Use half the available cores, capped at 8, to balance performance and memory for the JVM.
    # When running as a frozen executable, be conservative with memory.
    if getattr(sys, 'frozen', False):
        max_workers = 2
    else:
        # In a normal script environment, we can use more cores.
        max_workers = min(max(1, (os.cpu_count() or 1) // 2), 8)
    print(f"Using {max_workers} workers for Python feature analysis.")

    with ProcessPoolExecutor(max_workers=max_workers, initializer=partial(start_jvm, jvm_jars)) as executor:
        future_to_path = {executor.submit(analyze_python_features, path): path for path in crop_paths}
        for future in tqdm(as_completed(future_to_path), total=len(crop_paths), desc="Stage 3: Python Analysis"):
            result = future.result()
            if result:
                results.append(result)
    return pl.DataFrame([r for r in results if r])

def main():
    setup_directories()
    
    main_df = run_stage_1_ml_segmentation(INPUT_DIR_CUTS, COLUMN_DIR, MODEL_PATH, CROPS_DIR, MASKS_DIR)

    if main_df.is_empty():
        print("No fingerprints were segmented by the ML model. Exiting.")
        return

    print(f"\n--- Analyzing {main_df.height} successfully segmented fingerprints ---")
    
    cropped_image_paths = [
        os.path.join(CROPS_DIR, f) 
        for f in main_df.select("filename").to_series().to_list()
    ]

    jar_path = os.path.join(PROJECT_ROOT, "bin")
    jvm_jars = glob.glob(os.path.join(jar_path, "*.jar"))
    # Initialize dataframes for analysis results
    nfiq_df = pl.DataFrame()
    python_features_df = pl.DataFrame()

    if not jvm_jars:
        print("Warning: No JAR files found in 'bin/'. Skipping Minutiae analysis.")
        # nfiq_df is already an empty df
        python_features_df = pl.DataFrame() # Ensure it's an empty df
    else:
        nfiq_df = run_stage_2_nfiq2(cropped_image_paths)
        python_features_df = run_stage_3_python_analysis(cropped_image_paths, jvm_jars)

    print("\n--- Stage 4: Consolidating all results ---")
    
    final_df = main_df
    
    if not nfiq_df.is_empty():
        final_df = final_df.join(nfiq_df, on="filename", how="left")
        
    if not python_features_df.is_empty():
        final_df = final_df.join(python_features_df, on="filename", how="left")
    
    final_df.write_csv(FINAL_RESULTS_CSV)
    print(f"Processing complete. Final results saved to {FINAL_RESULTS_CSV}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()