import cv2
import numpy as np
import os

def merge_and_rotate_fingerprints(image_paths: list, fixed_height: int = 272):
    """
    Loads a list of fingerprint images, resizes each to a fixed height while
    maintaining aspect ratio, concatenates them horizontally into a single strip,
    and then rotates the strip 90 degrees clockwise to form a column.
    """
    if not image_paths:
        print("Warning: Received an empty list of image paths. Cannot create a column.")
        return None

    resized_images = []
    for path in image_paths:
        # Fix for Unicode paths on Windows
        try:
            with open(path, 'rb') as f:
                img_np = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error reading file {path} with buffer method: {e}")
            img = None # Fallback or ensure it's None
        if img is None:
            print(f"Warning: Could not read image at {path}. Skipping.")
            continue

        # The core resizing logic: resize height to a fixed value,
        # and scale width proportionally.
        h, w = img.shape[:2]
        scale = fixed_height / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, fixed_height), interpolation=cv2.INTER_AREA)
        resized_images.append(resized)

    if not resized_images:
        print("Error: No images could be read and resized. Aborting.")
        return None

    # Concatenate the resized images horizontally.
    # cv2.hconcat requires all images to have the same height, which our loop ensures.
    try:
        horizontal_strip = cv2.hconcat(resized_images)
    except cv2.error as e:
        print(f"Error during horizontal concatenation: {e}")
        print("This can happen if images have different numbers of channels (e.g., grayscale vs BGR).")
        return None

    # Rotate the final strip to create the vertical column.
    column_image = cv2.rotate(horizontal_strip, cv2.ROTATE_90_CLOCKWISE)

    return column_image