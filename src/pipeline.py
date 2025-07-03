import os
import glob
import subprocess
import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
from PIL import Image
import numpy as np

# Import feature analysis functions
from features.segmentation import segment_fingerprint
from features.minutiae import analyze_minutiae_from_image
from features.shape import analyze_shape
from features.texture import analyze_texture
from features.noise import analyze_background_noise
from features.frequency import analyze_ridge_frequency

# --- Configuration ---
INPUT_DIR = "input/bmp"
OUTPUT_DIR = "output"
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
NFIQ2_RESULTS_CSV = os.path.join(OUTPUT_DIR, "nfiq_results.csv")
FINAL_RESULTS_CSV = os.path.join(OUTPUT_DIR, "full_analysis.csv")
NFIQ2_EXECUTABLE_PATH = os.path.abspath("bin/NFIQ2/bin/NFIQ2.exe")

def setup_directories():
    os.makedirs(CROPS_DIR, exist_ok=True)
    
def save_crop_500dpi(arr: np.ndarray, dst: str) -> None:
    """
    Grava o array OpenCV BGR/Gray como BMP com metadado 500 dpi.
    """
    if arr.ndim == 2:                   # gray
        pil_img = Image.fromarray(arr, mode="L")
    else:                               # BGR → RGB
        pil_img = Image.fromarray(arr[:, :, ::-1], mode="RGB")
    pil_img.save(dst, format="BMP", dpi=(500, 500))

def run_stage_1_segmentation(image_paths):
    print("--- Stage 1: Segmenting and Cropping Images (Sequential) ---")
    results = []
    for i, path in enumerate(image_paths):
        try:
            segment_data = segment_fingerprint(path)
            if segment_data:
                filename = os.path.basename(path)
                if segment_data.get("cropped_image") is not None:
                    crop_path = os.path.join(CROPS_DIR, filename)
                    #cv2.imwrite(crop_path, segment_data["cropped_image"])
                    save_crop_500dpi(segment_data["cropped_image"], crop_path)
                
                results.append({
                    "filename": filename,
                    "is_single": segment_data["is_single"],
                    "box": str(segment_data["box"])
                })
            else:
                results.append({
                    "filename": os.path.basename(path), 
                    "is_single": False,
                    "box": None
                })
        except Exception as e:
            print(f"Error in Stage 1 for {os.path.basename(path)}: {e}")
        print(f"Progress: {i+1}/{len(image_paths)} complete.")
    return pl.DataFrame(results)


def run_stage_2_nfiq2():
    print("\n--- Stage 2: Running NFIQ2 Analysis (using simplified logic) ---")
    
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
        NFIQ2_EXECUTABLE_PATH,
        "-f", batch_file_path,
        "-o", temp_output_path,
        "-j", str(min(os.cpu_count() * 2, 8)),
        "-F"
    ]
    
    print(f"Executing: {' '.join(command)}")
    
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        print("NFIQ2 analysis complete.")
    except subprocess.CalledProcessError as e:
        print(f"NFIQ2 execution failed: {e.stderr.decode(errors='ignore')}")
        if os.path.exists(batch_file_path):
            os.remove(batch_file_path)
        return pl.DataFrame()

    if os.path.exists(batch_file_path):
        os.remove(batch_file_path)

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
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            
    return nfiq_df


def analyze_python_features(crop_path):
    """Function to run all Python-based analyses for a single cropped image."""
    try:
        filename = os.path.basename(crop_path)
        minutiae_data = analyze_minutiae_from_image(crop_path)
        cropped_image = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        original_path = os.path.join(INPUT_DIR, filename)
        original_img_gray = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        _, full_mask = cv2.threshold(cv2.GaussianBlur(original_img_gray, (5, 5), 2), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        shape_data = analyze_shape(cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
        texture_data = analyze_texture(cropped_image)
        noise_data = analyze_background_noise(full_mask)
        frequency_data = analyze_ridge_frequency(cropped_image)
        return {
            "filename": filename,
            **minutiae_data,
            **shape_data,
            **texture_data,
            **noise_data,
            **frequency_data
        }
    except Exception as e:
        print(f"Error in Python analysis for {os.path.basename(crop_path)}: {e}")
        return {"filename": os.path.basename(crop_path)}

def run_stage_3_python_analysis():
    print("\n--- Stage 3: Running Python-based Feature Analysis (Sequential) ---")
    crop_paths = glob.glob(os.path.join(CROPS_DIR, "*.bmp"))
    results = []
    for i, path in enumerate(crop_paths):
        try:
            result = analyze_python_features(path)
            results.append(result)
        except Exception as e:
            print(f"Critical error in Stage 3 for {os.path.basename(path)}: {e}")
        print(f"Progress: {i+1}/{len(crop_paths)} complete.")
    return pl.DataFrame(results)

def main():
    setup_directories()
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.bmp"))
    if not image_paths:
        print(f"No BMP images found in {INPUT_DIR}. Exiting.")
        return
    main_df = run_stage_1_segmentation(image_paths)
    if main_df.height == 0:
        print("No single fingerprints found. Exiting.")
        return
    nfiq_df = run_stage_2_nfiq2()
    python_features_df = run_stage_3_python_analysis()
    print("\n--- Stage 4: Consolidating all results ---")
    final_df = main_df.join(nfiq_df, on="filename", how="left")
    final_df = final_df.join(python_features_df, on="filename", how="left")
    final_df.write_csv(FINAL_RESULTS_CSV)
    print(f"\nProcessing complete. Final results saved to {FINAL_RESULTS_CSV}")

if __name__ == "__main__":
    main()
