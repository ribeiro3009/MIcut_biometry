import os
import sys
import multiprocessing
import polars as pl
import cv2
from tqdm import tqdm

from .segmentation import create_columns_from_cuts, segment_columns_with_ml
from .deep_ensemble import DeepEnsemble


def get_project_root():
  if getattr(sys, 'frozen', False):
        return sys._MEIPASS
  else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


PROJECT_ROOT = get_project_root()
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")
INPUT_DIR_CUTS = os.path.join(PROJECT_ROOT, "data", "input", "Fingerprints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")
COLUMN_DIR = os.path.join(OUTPUT_DIR, "merged_columns_from_pipeline")
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
MODEL_PATH = os.path.join(PROJECT_ROOT, "bin", "best_detector_model_v2.pth")
DEEP_RESULTS_CSV = os.path.join(OUTPUT_DIR, "deep_quality.csv")


def setup_directories():
    os.makedirs(CROPS_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(COLUMN_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "input", "Fingerprints"), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "bin"), exist_ok=True)
    os.makedirs(RESOURCES_DIR, exist_ok=True)


def compute_deep_quality_for_crops(crop_paths, device="cpu") -> pl.DataFrame:
    if not crop_paths:
        return pl.DataFrame()
    deep_q = DeepEnsemble(resources_dir=RESOURCES_DIR, device=device)
    rows = []
    for crop_path in tqdm(crop_paths, desc="DeepEnsemble: evaluating crops"):
        filename = os.path.basename(crop_path)
        image = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            rows.append({"filename": filename})
            continue
        try:
            preds = deep_q.predict_ensemble(image)
            fused = deep_q.fusion(preds)
            rows.append({
                "filename": filename,
                "vfq": int(preds.get("vfq", 0)),
                "nfq": int(preds.get("nfq", 0)),
                "lqm": int(preds.get("lqm", 0)),
                "mor": int(preds.get("mor", 0)),
                "fused": int(fused),
            })
        except Exception as e:
            rows.append({"filename": filename, "error": str(e)})
    return pl.DataFrame(rows)


def main():
    setup_directories()

    print("--- Stage 1: Creating Columns and Segmenting with ML Model ---")
    column_paths = create_columns_from_cuts(INPUT_DIR_CUTS, COLUMN_DIR)
    if not column_paths:
        print("No columns were created. Exiting.")
        return
    print(f"Successfully created {len(column_paths)} column images.")

    main_df = segment_columns_with_ml(column_paths, MODEL_PATH, CROPS_DIR, MASKS_DIR)
    if main_df.is_empty():
        print("No fingerprints were segmented by the ML model. Exiting.")
        return

    print(f"\n--- Deep learning quality on {main_df.height} segmented fingerprints ---")
    cropped_image_paths = [
        os.path.join(CROPS_DIR, f) 
        for f in main_df.select("filename").to_series().to_list()
    ]
    deep_df = compute_deep_quality_for_crops(cropped_image_paths, device="cpu")

    final_df = main_df
    if not deep_df.is_empty():
        final_df = final_df.join(deep_df, on="filename", how="left")

    final_df.write_csv(DEEP_RESULTS_CSV)
    print(f"Processing complete. Deep quality results saved to {DEEP_RESULTS_CSV}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()


