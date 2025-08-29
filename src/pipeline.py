import os
import glob
import sys
import subprocess
import multiprocessing
import time
import polars as pl
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
from functools import partial
import cbor2
import jpype, jpype.imports

# Import feature analysis functions
from features.ml_segmentation import create_columns_from_cuts, segment_columns_with_ml
from features.minutiae import start_jvm, extract_fingerprint_template, detect_clusters_and_singularities
from features.shape import analyze_shape
from features.texture import analyze_texture
from features.frequency import analyze_ridge_frequency
from features.dfiqi import dfiqi_on_image, DFIQIParams

# --- Configuration ---
def get_project_root():
    """ Retorna o caminho raiz do projeto, seja rodando como script ou como executável. """
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
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
    start_time = time.time()
    print("--- Stage 1: Creating Columns and Segmenting with ML Model ---")
    print(f"Creating column images from cuts in '{input_dir}'...")
    column_paths = create_columns_from_cuts(input_dir, column_dir)
    if not column_paths:
        print("No columns were created. Aborting Stage 1.")
        return pl.DataFrame(), 0
    print(f"Successfully created {len(column_paths)} column images.")
    result_df = segment_columns_with_ml(column_paths, model_path, crops_dir, masks_dir)
    duration = time.time() - start_time
    return result_df, duration

def run_stage_2_nfiq2(crop_files: list[str], num_cores: int):
    start_time = time.time()
    print(f"\n--- Stage 2: Running NFIQ2 Analysis (using {num_cores} cores) ---")
    root_dir = os.getcwd()
    batch_file_path = os.path.join(root_dir, "nfiq2_batch.txt")
    temp_output_path = os.path.join(root_dir, "nfiq2_output.csv")
    
    if not crop_files:
        print("No cropped images found to analyze with NFIQ2.")
        return pl.DataFrame(), 0

    with open(batch_file_path, 'w') as f:
        for path in crop_files:
            f.write(f"{os.path.abspath(path)}\n")

    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    command = [
        NFIQ2_EXECUTABLE_PATH, "-f", batch_file_path, "-o", temp_output_path,
        "-j", str(num_cores), "-F"
    ]
    
    try:
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
    except subprocess.CalledProcessError as e:
        print(f"NFIQ2 execution failed: {e.stderr.decode(errors='ignore') if e.stderr else 'Unknown error'}")
        if os.path.exists(batch_file_path): os.remove(batch_file_path)
        return pl.DataFrame(), time.time() - start_time

    if os.path.exists(batch_file_path): os.remove(batch_file_path)

    if not os.path.exists(temp_output_path):
        print("NFIQ2 did not produce an output file.")
        return pl.DataFrame(), time.time() - start_time

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
            
    duration = time.time() - start_time
    print(f"NFIQ2 analysis complete in {duration:.2f}s.")
    return nfiq_df, duration

def _imread_unicode(path, flags):
    """cv2.imread wrapper for unicode paths."""
    try:
        with open(path, 'rb') as f:
            img_np = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(img_np, flags)
        return img
    except (IOError, FileNotFoundError):
        return None

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

        template_cbor = extract_fingerprint_template(cropped_image, dpi=500)
        minutiae_data = detect_clusters_and_singularities(template_cbor)

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

        shape_data = analyze_shape(roi_mask)
        texture_data = analyze_texture(cropped_image, roi_mask)
        frequency_data = analyze_ridge_frequency(cropped_image, roi_mask)

        dfiqi_fields = {}
        try:
            if features_xy:
                dfiqi_out = dfiqi_on_image(cropped_image, features_xy, DFIQIParams(dpi=500.0))
                g = dfiqi_out["global"]
                dfiqi_fields = {
                    "DFIQI_LQSsum": g["LQSsum"],
                    "DFIQI_ValueGQS": g["ValueGQS"],
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
    except Exception as e:
        print(f"Error in Python analysis for {os.path.basename(crop_path)}: {e}")
        return {"filename": os.path.basename(crop_path), "error": str(e)}

def run_stage_3_python_analysis(crop_paths: list[str], jvm_jars: list[str], max_workers: int):
    start_time = time.time()
    print(f"--- Stage 3: Running Python-based Feature Analysis (using {max_workers} workers) ---")
    if not crop_paths:
        print("No cropped images found for Python analysis.")
        return pl.DataFrame(), 0
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=partial(start_jvm, jvm_jars)) as executor:
        future_to_path = {executor.submit(analyze_python_features, path): path for path in crop_paths}
        for future in tqdm(as_completed(future_to_path), total=len(crop_paths), desc="Stage 3: Python Analysis"):
            result = future.result()
            if result:
                results.append(result)
    
    duration = time.time() - start_time
    print(f"Python-based feature analysis complete in {duration:.2f}s.")
    return pl.DataFrame([r for r in results if r]), duration

def main():
    pipeline_start_time = time.time()
    timings = {}

    setup_directories()
    
    main_df, stage1_duration = run_stage_1_ml_segmentation(INPUT_DIR_CUTS, COLUMN_DIR, MODEL_PATH, CROPS_DIR, MASKS_DIR)
    timings["Stage 1: ML Segmentation"] = stage1_duration

    if main_df.is_empty():
        print("No fingerprints were segmented by the ML model. Exiting.")
        return

    print(f"\n--- Analyzing {main_df.height} successfully segmented fingerprints ---")
    
    cropped_image_paths = [os.path.join(CROPS_DIR, f) for f in main_df.select("filename").to_series().to_list()]
    jar_path = os.path.join(PROJECT_ROOT, "bin")
    jvm_jars = glob.glob(os.path.join(jar_path, "*.jar"))

    nfiq_df = pl.DataFrame()
    python_features_df = pl.DataFrame()

    if not jvm_jars:
        print("Warning: No JAR files found in 'bin/'. Skipping Minutiae and NFIQ2 analysis.")
    else:
        total_cores = os.cpu_count() or 1
        nfiq_cores = max(1, total_cores // 2)
        python_workers = max(1, total_cores - nfiq_cores)
        
        print(f"\n--- Starting concurrent analysis (Total Cores: {total_cores}) ---")
        print(f"    - NFIQ2 allocated {nfiq_cores} cores.")
        print(f"    - Python/SourceAFIS allocated {python_workers} workers.")

        concurrent_start_time = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_nfiq = executor.submit(run_stage_2_nfiq2, cropped_image_paths, num_cores=nfiq_cores)
            future_py_features = executor.submit(run_stage_3_python_analysis, cropped_image_paths, jvm_jars, max_workers=python_workers)
            
            nfiq_df, timings["Stage 2: NFIQ2 (individual)"] = future_nfiq.result()
            python_features_df, timings["Stage 3: Python/SourceAFIS (individual)"] = future_py_features.result()
        timings["Concurrent Stages 2 & 3 (Wall Time)"] = time.time() - concurrent_start_time

    consolidation_start_time = time.time()
    print("\n--- Stage 4: Consolidating all results ---")
    
    final_df = main_df
    
    if not nfiq_df.is_empty():
        final_df = final_df.join(nfiq_df, on="filename", how="left")
        
    if not python_features_df.is_empty():
        final_df = final_df.join(python_features_df, on="filename", how="left")
    
    final_df.write_csv(FINAL_RESULTS_CSV)
    timings["Stage 4: Consolidation"] = time.time() - consolidation_start_time

    total_pipeline_time = time.time() - pipeline_start_time
    timings["Total Pipeline"] = total_pipeline_time

    print("\n--- BENCHMARK RESULTS ---")
    for stage, duration in timings.items():
        print(f"{stage:<40}: {duration:.2f} seconds")
    print("-------------------------")
    print(f"Processing complete. Final results saved to {FINAL_RESULTS_CSV}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()