
import numpy as np
import cv2

def analyze_ridge_frequency(cropped_image):
    """
    Estimates the ridge frequency of a fingerprint image.

    Args:
        cropped_image (numpy.ndarray): The cropped fingerprint image.

    Returns:
        dict: A dictionary with 'ridge_frequency'.
    """
    if cropped_image is None:
        return {"ridge_frequency": 0.0}

    try:
        # Normalize the image
        normalized_image = cv2.normalize(cropped_image, None, 0, 255, cv2.NORM_MINMAX)

        # Perform FFT
        f = np.fft.fft2(normalized_image)
        fshift = np.fft.fftshift(f)

        # Find the peak frequency
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        
        # We look for the peak in a region away from the center (DC component)
        # This is a simplified approach; more robust methods exist but are complex.
        # For this, we'll find the location of the max value in the spectrum.
        max_loc = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
        
        # Calculate frequency from the distance to the center
        freq = np.sqrt((max_loc[0] - center_y)**2 + (max_loc[1] - center_x)**2)

        return {"ridge_frequency": freq}

    except Exception as e:
        print(f"Error analyzing ridge frequency: {e}")
        return {"ridge_frequency": 0.0}
