
import cv2
import numpy as np

def analyze_background_noise(full_image_mask):
    """
    Calculates the percentage of noise outside the main fingerprint area.

    Args:
        full_image_mask (numpy.ndarray): The binary mask of the entire original image.

    Returns:
        dict: A dictionary with 'bg_noise'.
    """
    default_return = {"bg_noise": 0.0}
    if full_image_mask is None or not np.any(full_image_mask):
        return default_return

    try:
        # Find the largest contour which represents the fingerprint
        contours, _ = cv2.findContours(full_image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return default_return

        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the convex hull of the fingerprint
        hull = cv2.convexHull(largest_contour)
        hull_mask = np.zeros_like(full_image_mask)
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

        # Invert the hull mask to get the area *outside* the fingerprint
        outside_hull_mask = cv2.bitwise_not(hull_mask)

        # Calculate noise by finding pixels from the original mask that are outside the hull
        noise = cv2.bitwise_and(full_image_mask, outside_hull_mask)

        # Calculate the ratio of noise pixels to the total pixels in the mask
        noise_pixels = np.count_nonzero(noise)
        total_pixels = np.count_nonzero(full_image_mask)

        bg_noise_ratio = float(noise_pixels) / total_pixels if total_pixels > 0 else 0.0

        return {"bg_noise": bg_noise_ratio}

    except Exception as e:
        print(f"Error analyzing background noise: {e}")
        return default_return
