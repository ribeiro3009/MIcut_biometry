from PIL import Image
import os
from glob import glob

INPUT_DIR = "C:/Users/rj0369870548/Desktop/Projects_VsStudio/Recorte_Library/MIcut_biometric/output/crops"
OUTPUT_DIR = "C:/bmp_resized"
MAX_WIDTH = 800

os.makedirs(OUTPUT_DIR, exist_ok=True)

for path in glob(f"{INPUT_DIR}/*.bmp"):
    img = Image.open(path)
    w, h = img.size
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        new_size = (MAX_WIDTH, int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
    
    filename = os.path.basename(path)
    output_path = os.path.join(OUTPUT_DIR, filename)
    img.save(output_path, dpi=(500, 500))
