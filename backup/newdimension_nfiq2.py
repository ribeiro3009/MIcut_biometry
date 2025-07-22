from PIL import Image
import os
from glob import glob

INPUT_DIR = "C:/Users/rj0369870548/Desktop/Projects_VsStudio/Recorte_Library/MIcut_biometric/input/bmp"
OUTPUT_DIR = "C:/bmp_resized_dpi"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for path in glob(f"{INPUT_DIR}/*.bmp"):
    img = Image.open(path)

    filename = os.path.basename(path)
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Regrava imagem com DPI 500x500
    img.save(output_path, dpi=(500, 500))
