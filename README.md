# MICUT Biometry Deep

Standalone deep-learning quality pipeline extracted from the main project. It performs:

- Column creation from input fingerprint cuts
- ML-based segmentation to detect finger boxes and save crops + masks
- DeepEnsemble inference to compute vfq, nfq, lqm, mor and fused score
- CSV export with filenames, box coordinates and scores

## Folder layout

- `micut_deep/`: Python package (pipeline, segmentation, DeepEnsemble)
- `bin/`: Place model files here:
  - `best_detector_model_v2.pth`
- `resources/`: Place model files here:
  - `model_densenet121.pt`
  - `pca_fusion_model.pkl`
- `data/input/Fingerprints/`: Put BMP cuts here (e.g., `123_dedo1.bmp`, ...)
- `data/output/`: Outputs go here:
  - `deep_quality.csv`
  - `crops/`, `masks/`, `merged_columns_from_pipeline/`

## Setup

1) Create and activate a virtual environment in this folder (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install requirements:

```powershell
pip install -r requirements.txt
```

3) Place model resources in the `bin` and `resources` directories as described in the "Folder layout" section.

## Run

From this `MICUT_Biometry_Deep` folder:

```powershell
python -m micut_deep.pipeline
```

Results will be saved to `data/output/deep_quality.csv`.