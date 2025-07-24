import os
import glob
import subprocess
import polars as pl
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
from functools import partial

# Import feature analysis functions
from features.segmentation import segment_fingerprint
from features.minutiae import analyze_minutiae_from_image, start_jvm
from features.shape import analyze_shape
from features.texture import analyze_texture
from features.frequency import analyze_ridge_frequency

# --- Configuration ---
INPUT_DIR = "input/bmp"
OUTPUT_DIR = "output"
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
NFIQ2_RESULTS_CSV = os.path.join(OUTPUT_DIR, "nfiq_results.csv")
FINAL_RESULTS_CSV = os.path.join(OUTPUT_DIR, "full_analysis.csv")
NFIQ2_EXECUTABLE_PATH = os.path.abspath("bin/NFIQ2/bin/NFIQ2.exe")

def setup_directories():
    os.makedirs(CROPS_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
    
def save_crop_500dpi(arr: np.ndarray, dst: str) -> None:
    """
    Grava o array OpenCV BGR/Gray como BMP com metadado 500 dpi.
    """
    if arr.ndim == 2:                   # gray
        pil_img = Image.fromarray(arr, mode="L")
    else:                               # BGR → RGB
        pil_img = Image.fromarray(arr[:, :, ::-1], mode="RGB")
    pil_img.save(dst, format="BMP", dpi=(500, 500))

def process_single_image_for_segmentation(path):
    """Helper function for parallel execution of segmentation."""
    try:
        orig_bgr = cv2.imread(path)
        if orig_bgr is None:
            print(f"Error: Could not read image {os.path.basename(path)}")
            return None
        segment_data = segment_fingerprint(orig_bgr)
        if segment_data and segment_data.get("box") is not None:
            filename = os.path.basename(path)
            if segment_data.get("cropped_image") is not None:
                crop_path = os.path.join(CROPS_DIR, filename)
                save_crop_500dpi(segment_data["cropped_image"], crop_path)
            
            if segment_data.get("mask") is not None:
                mask_path = os.path.join(MASKS_DIR, filename.replace(".bmp", ".png"))
                cv2.imwrite(mask_path, segment_data["mask"].astype(np.uint8))

            return {
                "filename": filename,
                "is_single": segment_data["is_single"],
                "box_x1": segment_data["box"][0],
                "box_y1": segment_data["box"][1],
                "box_x2": segment_data["box"][2],
                "box_y2": segment_data["box"][3],
            }
        else:
            return {
                "filename": os.path.basename(path), "is_single": False, 
                "box_x1": None, "box_y1": None, "box_x2": None, "box_y2": None,
            }
    except Exception as e:
        print(f"Error processing {os.path.basename(path)} in Stage 1: {e}")
        return None

def run_stage_1_segmentation(image_paths):
    print("--- Stage 1: Segmenting and Cropping Images (Parallel) ---")
    results = []
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() // 2, 4)) as executor:
        future_to_path = {executor.submit(process_single_image_for_segmentation, path): path for path in image_paths}
        for future in tqdm(as_completed(future_to_path), total=len(image_paths), desc="Stage 1: Segmentation"):
            result = future.result()
            if result:
                results.append(result)
    
    return pl.DataFrame([r for r in results if r])

def run_stage_2_nfiq2(crop_files: list[str]):
    print("\n--- Stage 2: Running NFIQ2 Analysis ---")
    
    root_dir = os.getcwd()
    batch_file_path = os.path.join(root_dir, "nfiq2_batch.txt")
    temp_output_path = os.path.join(root_dir, "nfiq2_output.csv")
    
    crop_files = glob.glob(os.path.join(CROPS_DIR, "*.bmp"))
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
        print(f"NFIQ2 execution failed: {e.stderr.decode(errors='ignore')}")
        if os.path.exists(batch_file_path): os.remove(batch_file_path)
        return pl.DataFrame()

    if os.path.exists(batch_file_path): os.remove(batch_file_path)

    if not os.path.exists(temp_output_path):
        print("NFIQ2 did not produce an output file.")
        return pl.DataFrame()

    try:
        nfiq_df = pl.read_csv(temp_output_path, encoding='latin-1').rename({"Filename": "filename", "QualityScore": "NFIQ2"})
        nfiq_df = nfiq_df.select(["filename", "NFIQ2"])
        nfiq_df = nfiq_df.with_columns(
            pl.col("filename").map_elements(lambda x: os.path.basename(x.strip('"')), return_dtype=pl.Utf8)
        )
    finally:
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
            
    return nfiq_df

def analyze_python_features(crop_path):
    """Function to run all Python-based analyses for a single cropped image."""
    try:
        filename = os.path.basename(crop_path)
        
        mask_path = os.path.join(MASKS_DIR, filename.replace(".bmp", ".png"))
        roi_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if roi_mask is not None: # Garante que a máscara é binária (0 ou 1)
            roi_mask = (roi_mask > 0).astype(np.uint8)

        cropped_image = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if cropped_image is None: return {"filename": filename}

        minutiae_data = analyze_minutiae_from_image(cropped_image)
        shape_data = analyze_shape(roi_mask)
        texture_data = analyze_texture(cropped_image, roi_mask)
        frequency_data = analyze_ridge_frequency(cropped_image, roi_mask)

        return {"filename": filename, **minutiae_data, **shape_data, **texture_data, **frequency_data}
    except Exception as e:
        print(f"Error in Python analysis for {os.path.basename(crop_path)}: {e}")
        return {"filename": os.path.basename(crop_path), "error": str(e)}

def run_stage_3_python_analysis(crop_paths: list[str], jvm_jars: list[str]):
    print("--- Stage 3: Running Python-based Feature Analysis (Parallel) ---")
    if not crop_paths:
        print("No cropped images found for Python analysis.")
        return pl.DataFrame()
    results = []
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() // 2, 4), initializer=partial(start_jvm, jvm_jars)) as executor:
        future_to_path = {executor.submit(analyze_python_features, path): path for path in crop_paths}
        for future in tqdm(as_completed(future_to_path), total=len(crop_paths), desc="Stage 3: Python Analysis"):
            result = future.result()
            if result:
                results.append(result)
    return pl.DataFrame([r for r in results if r])

def main():
    setup_directories()
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.bmp"))
    if not image_paths:
        print(f"No BMP images found in {INPUT_DIR}. Exiting.")
        return

    main_df = run_stage_1_segmentation(image_paths)
    if main_df.filter(pl.col("box_x1").is_not_null()).height == 0:
        print("Stage 1 did not produce any valid segmentations. Exiting.")
        return

    print("\n--- Running Stage 2 (NFIQ2) and Stage 3 (Python Analysis) sequentially ---")
    
    cropped_image_paths = [os.path.join(CROPS_DIR, f) for f in main_df.select("filename").to_series().to_list()]

    jar_path = os.path.abspath("bin")
    jvm_jars = glob.glob(os.path.join(jar_path, "*.jar"))
    if not jvm_jars:
        print("Nenhum arquivo JAR encontrado em 'bin/'. Garanta que o SourceAFIS está lá. Exiting.")
        return

    nfiq_df = run_stage_2_nfiq2(cropped_image_paths)
    python_features_df = run_stage_3_python_analysis(cropped_image_paths, jvm_jars)

    print("\n--- Stage 4: Consolidating all results ---")
    
    final_df = main_df.join(nfiq_df, on="filename", how="left")
    if python_features_df.height > 0:
        final_df = final_df.join(python_features_df, on="filename", how="left")
    
    final_df.write_csv(FINAL_RESULTS_CSV)
    print(f"Processing complete. Final results saved to {FINAL_RESULTS_CSV}")

if __name__ == "__main__":
    main()