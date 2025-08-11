
import os
import glob
import cv2
from collections import defaultdict
from .merge_column import merge_and_rotate_fingerprints

# --- Main Functions ---

def create_columns_from_cuts(input_dir: str, output_dir: str) -> list:
    os.makedirs(output_dir, exist_ok=True)
    all_image_paths = glob.glob(os.path.join(input_dir, "*.bmp"))
    images_by_id = defaultdict(list)

    for path in all_image_paths:
        filename = os.path.basename(path)
        try:
            img_id = filename.split('_')[0]
            images_by_id[img_id].append(path)
        except IndexError:
            print(f"Warning: File {filename} does not match 'ID_dedoX.bmp' format. Skipping.")
            continue

    if not images_by_id:
        print(f"Error: No valid BMP images found in {input_dir}.")
        return []

    created_column_paths = []
    for img_id, paths in images_by_id.items():
        try:
            paths.sort(key=lambda p: int(os.path.basename(p).split('_dedo')[1].split('.')[0]))
        except (IndexError, ValueError):
            print(f"Warning: Could not sort files for ID {img_id}. Check file naming. Skipping.")
            continue

        hand1_paths = [p for p in paths if int(os.path.basename(p).split('_dedo')[1].split('.')[0]) <= 5]
        hand2_paths = [p for p in paths if int(os.path.basename(p).split('_dedo')[1].split('.')[0]) > 5]

        if hand1_paths:
            output_path = process_and_save_hand(img_id, hand1_paths, "hand1", output_dir)
            if output_path:
                created_column_paths.append(output_path)

        if hand2_paths:
            output_path = process_and_save_hand(img_id, hand2_paths, "hand2", output_dir)
            if output_path:
                created_column_paths.append(output_path)
                
    return created_column_paths

def process_and_save_hand(img_id, hand_paths, hand_name, output_dir):
    column_image = merge_and_rotate_fingerprints(hand_paths)
    
    # --- DEBUG: Check if the column image was created ---
    if column_image is None:
        print(f"Error: merge_and_rotate_fingerprints returned None for ID {img_id}, {hand_name}.")
        return None
    # --- END DEBUG ---

    output_filename = f"column_{img_id}_{hand_name}.png"
    output_path = os.path.normpath(os.path.join(output_dir, output_filename))
    
    try:
        # Encode the image to a memory buffer
        is_success, buffer = cv2.imencode(".png", column_image)
        if not is_success:
            raise IOError("cv2.imencode failed to encode image.")
        
        # Write the buffer to file using Python's built-in I/O
        with open(output_path, 'wb') as f:
            f.write(buffer)
            
    except Exception as e:
        print(f"Error saving column image for ID {img_id}, {hand_name}: {e}")
        return None

    # --- DEBUG: Verify file existence after writing ---
    if not os.path.exists(output_path):
        print(f"FATAL: cv2.imwrite reported success, but file does not exist at {output_path}")
        return None
    # --- END DEBUG ---

    return output_path

def segment_columns_with_ml(column_paths, model_path, crops_dir, masks_dir, batch_size=4, threshold=0.8):
    import torch
    import torchvision.transforms.functional as F
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    import numpy as np
    import polars as pl
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # --- Utility Functions (nested or local) ---
    def get_model(num_classes):
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def create_batches(items, batch_size):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def save_crop_500dpi(arr: np.ndarray, dst: str) -> None:
        from PIL import Image
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr, mode="L")
        else:
            pil_img = Image.fromarray(arr[:, :, ::-1], mode="RGB")
        pil_img.save(dst, format="BMP", dpi=(500, 500))

    def _imwrite_unicode(path, img):
        """cv2.imwrite wrapper for unicode paths."""
        try:
            is_success, buffer = cv2.imencode(os.path.splitext(path)[1], img)
            if not is_success:
                raise IOError(f"cv2.imencode failed for {path}")
            with open(path, 'wb') as f:
                f.write(buffer)
            return True
        except Exception:
            return False

    def remove_lines_keep_fingerprints(img: np.ndarray):
        """Remove linhas verticais/horizontais mantendo as cristas das digitais."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        all_lines = cv2.bitwise_or(vertical_lines, horizontal_lines)
        fingerprints_only = cv2.bitwise_and(thresh, cv2.bitwise_not(all_lines))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(fingerprints_only, cv2.MORPH_OPEN, kernel_small)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filtered = cv2.dilate(cleaned, kernel_dilate, iterations=1)
        return filtered

    def process_and_save_crop(col_img_gray, box, person_id, finger_index, crops_dir, masks_dir):
        """Processes and saves a single cropped fingerprint and its mask."""
        x1, y1, x2, y2 = box
        output_filename = f"{person_id}_dedo{finger_index}.bmp"
        
        cropped_finger = col_img_gray[y1:y2, x1:x2]
        cropped_finger = cv2.rotate(cropped_finger, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        mask = remove_lines_keep_fingerprints(cropped_finger)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.dilate(closed, k, iterations=1)

        crop_path = os.path.join(crops_dir, output_filename)
        mask_path = os.path.join(masks_dir, output_filename.replace(".bmp", ".png"))
        
        save_crop_500dpi(cropped_finger, crop_path)
        _imwrite_unicode(mask_path, mask)

        return {
            "filename": output_filename, "is_single": True,
            "box_x1": x1, "box_y1": y1, "box_x2": x2, "box_y2": y2,
        }

    # --- Main Logic ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device} for ML inference.")

    model = get_model(num_classes=2)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Error loading model: {e}. Aborting segmentation.")
        return pl.DataFrame()
    model.to(device)
    model.eval()

    all_results = []
    tasks = []

    with ThreadPoolExecutor() as executor:
        for batch_paths in tqdm(create_batches(column_paths, batch_size), total=-(-len(column_paths) // batch_size), desc="Stage 1: ML Segmentation"):
            images_tensors = []
            original_column_images = []
            for path in batch_paths:
                with open(path, 'rb') as f:
                    img_np = np.frombuffer(f.read(), np.uint8)
                    col_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                images_tensors.append(F.to_tensor(cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB)).to(device))
                original_column_images.append(col_img)

            with torch.no_grad():
                predictions = model(images_tensors)

            for i, pred in enumerate(predictions):
                col_path = batch_paths[i]
                col_img = original_column_images[i]
                col_filename = os.path.basename(col_path)
                person_id, hand = col_filename.replace("column_", "").replace(".png", "").split('_')
                
                boxes = pred['boxes'][pred['scores'] > threshold].detach().cpu().numpy().astype(int)
                boxes = sorted(boxes, key=lambda b: b[1])

                finger_num_start = 1 if hand == "hand1" else 6
                gray_col = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)

                for j, box in enumerate(boxes):
                    finger_index = finger_num_start + j
                    # Submit task to the thread pool
                    future = executor.submit(process_and_save_crop, gray_col, box, person_id, finger_index, crops_dir, masks_dir)
                    tasks.append(future)
        
        # Collect results as they complete
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Saving crops and masks"):
            all_results.append(future.result())

    return pl.DataFrame(all_results)
