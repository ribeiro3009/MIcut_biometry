
import cv2
import numpy as np

def analyze_shape(mask):
    """
    Calculates shape-based metrics (solidity and coverage) from a binary mask.

    Args:
        mask (numpy.ndarray): The binary mask of the fingerprint.

    Returns:
        dict: A dictionary with 'solidity' and 'coverage'.
              Returns default values if mask is invalid or has no contour.
    """
    default_return = {"solidity": 0.0, "coverage": 0.0}
    if mask is None or not np.any(mask):
        return default_return

    try:
        # Find the largest contour from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return default_return

        largest_contour = max(contours, key=cv2.contourArea)

        # 1. Solidity
        area = cv2.contourArea(largest_contour)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        # 2. Coverage
        _, _, w, h = cv2.boundingRect(largest_contour)
        bounding_box_area = w * h
        coverage = float(area) / bounding_box_area if bounding_box_area > 0 else 0.0

        return {
            "solidity": solidity,
            "coverage": coverage
        }

    except Exception as e:
        print(f"Error analyzing shape: {e}")
        return default_return
