
import cv2
import numpy as np

def analyze_texture(cropped_image):
    """
    Calculates texture-based metrics: sharpness (Laplacian variance),
    orientation standard deviation, and local contrast.

    Args:
        cropped_image (numpy.ndarray): The cropped fingerprint image.

    Returns:
        dict: A dictionary with 'sharpness', 'orientation_std', and 'contrast'.
    """
    default_return = {"sharpness": 0.0, "orientation_std": 0.0, "contrast": 0.0}
    if cropped_image is None:
        return default_return

    try:
        # 1. Sharpness (Laplacian Variance)
        sharpness = cv2.Laplacian(cropped_image, cv2.CV_64F).var()

        # 2. Orientation Standard Deviation
        gx = cv2.Sobel(cropped_image, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(cropped_image, cv2.CV_32F, 0, 1)
        orientation_rad = np.arctan2(gy, gx)
        orientation_deg = np.degrees(orientation_rad)
        orientation_std = np.std(orientation_deg)

        # 3. Contrast (Root Mean Square contrast)
        contrast = np.sqrt(np.mean((cropped_image - np.mean(cropped_image))**2))

        return {
            "sharpness": sharpness,
            "orientation_std": orientation_std,
            "contrast": contrast
        }

    except Exception as e:
        print(f"Error analyzing texture: {e}")
        return default_return
