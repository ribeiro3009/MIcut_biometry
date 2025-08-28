import os
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
from typing import List
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def merge_and_rotate_fingerprints(image_paths: List[str], fixed_height: int = 272):
    if not image_paths:
        return None
    resized_images = []
    for path in image_paths:
        try:
            with open(path, 'rb') as f:
                img_np = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = fixed_height / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, fixed_height), interpolation=cv2.INTER_AREA)
        resized_images.append(resized)
    if not resized_images:
        return None
    try:
        horizontal_strip = cv2.hconcat(resized_images)
    except cv2.error:
        return None
    column_image = cv2.rotate(horizontal_strip, cv2.ROTATE_90_CLOCKWISE)
    return column_image


def create_columns_from_cuts(input_dir: str, output_dir: str) -> list:
    os.makedirs(output_dir, exist_ok=True)
    all_image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".bmp")]
    images_by_id = defaultdict(list)
    for path in all_image_paths:
        filename = os.path.basename(path)
        try:
            img_id = filename.split('_')[0]
            images_by_id[img_id].append(path)
        except IndexError:
            continue
    tasks = []
    for img_id, paths in images_by_id.items():
        try:
            paths.sort(key=lambda p: int(os.path.basename(p).split('_dedo')[1].split('.')[0]))
        except (IndexError, ValueError):
            continue
        hand1_paths = [p for p in paths if int(os.path.basename(p).split('_dedo')[1].split('.')[0]) <= 5]
        hand2_paths = [p for p in paths if int(os.path.basename(p).split('_dedo')[1].split('.')[0]) > 5]
        if hand1_paths:
            tasks.append((img_id, hand1_paths, "hand1", output_dir))
        if hand2_paths:
            tasks.append((img_id, hand2_paths, "hand2", output_dir))

    created_column_paths = []
    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) * 4)) as executor:
        futures = {executor.submit(process_and_save_hand, *task): task for task in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Creating column images"):
            result_path = fut.result()
            if result_path:
                created_column_paths.append(result_path)
    return created_column_paths


def process_and_save_hand(img_id, hand_paths, hand_name, output_dir):
    column_image = merge_and_rotate_fingerprints(hand_paths)
    if column_image is None:
        return None
    output_filename = f"column_{img_id}_{hand_name}.png"
    output_path = os.path.normpath(os.path.join(output_dir, output_filename))
    is_success, buffer = cv2.imencode(".png", column_image)
    if not is_success:
        return None
    with open(output_path, 'wb') as f:
        f.write(buffer)
    return output_path


def segment_columns_with_ml(column_paths, model_path, crops_dir, masks_dir, batch_size=4, threshold=0.8):
    import torch
    import torchvision.transforms.functional as F
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    def get_model(num_classes):
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def create_batches(items, batch_size):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def save_crop_500dpi(arr: np.ndarray, dst: str) -> None:
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr, mode="L")
        else:
            pil_img = Image.fromarray(arr[:, :, ::-1], mode="RGB")
        pil_img.save(dst, format="BMP", dpi=(500, 500))

    def _imwrite_unicode(path, img):
        try:
            is_success, buffer = cv2.imencode(os.path.splitext(path)[1], img)
            if not is_success:
                return False
            with open(path, 'wb') as f:
                f.write(buffer)
            return True
        except Exception:
            return False

    def remove_lines_keep_fingerprints(img: np.ndarray):
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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes=2)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return pl.DataFrame()
    except Exception as e:
        print(f"Error loading model: {e}")
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
                    future = executor.submit(process_and_save_crop, gray_col, box, person_id, finger_index, crops_dir, masks_dir)
                    tasks.append(future)
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Saving crops and masks"):
            all_results.append(future.result())
    return pl.DataFrame(all_results)


