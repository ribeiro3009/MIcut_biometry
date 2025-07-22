import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

def analyze_ridge_frequency(cropped_image, roi_mask=None, block_size=32, k=50.0):
    """
    Calculates ridge consistency score based on the standard deviation of local ridge frequencies.
    A score of 1.0 is perfect consistency, approaching 0.0 for high variance.

    Args:
        cropped_image (numpy.ndarray): The cropped grayscale fingerprint image.
        roi_mask (numpy.ndarray, optional): A binary mask of the region of interest.
                                            If provided, only blocks within the mask are analyzed.
        block_size (int): The size of the blocks to analyze.
        k (float): A scaling factor for the exponential decay function.

    Returns:
        dict: A dictionary with 'ridge_consistency'.
    """
    default_return = {"ridge_consistency": 0.0}
    if cropped_image is None:
        return default_return

    try:
        frequencies = []
        rows, cols = cropped_image.shape

        # Pre-calculate orientations using Sobel operators
        sobel_x = cv2.Sobel(cropped_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(cropped_image, cv2.CV_64F, 0, 1, ksize=5)
        
        gxx = cv2.GaussianBlur(sobel_x**2, (5, 5), 0)
        gyy = cv2.GaussianBlur(sobel_y**2, (5, 5), 0)
        gxy = cv2.GaussianBlur(sobel_x * sobel_y, (5, 5), 0)
        
        denom = gxx - gyy
        numer = 2 * gxy
        
        orientation_field = (np.pi + np.arctan2(numer, denom)) / 2

        for r in range(0, rows - block_size, block_size):
            for c in range(0, cols - block_size, block_size):
                
                if roi_mask is not None:
                    roi_block = roi_mask[r:r+block_size, c:c+block_size]
                    if np.mean(roi_block) < 255 * 0.25: # Analyze only if >25% is fingerprint
                        continue

                block = cropped_image[r:r+block_size, c:c+block_size]
                block_orientation = orientation_field[r + block_size//2, c + block_size//2]

                # Rotate block so ridges are vertical
                rot_mat = cv2.getRotationMatrix2D((block_size/2, block_size/2), np.degrees(block_orientation) + 90, 1)
                rotated_block = cv2.warpAffine(block, rot_mat, (block_size, block_size))
                
                # Project pixels to a 1D signal
                projection = np.sum(rotated_block, axis=0)
                
                # Get frequency from the 1D signal using FFT
                fft_proj = np.abs(np.fft.fft(projection))
                peak_index = np.argmax(fft_proj[1:block_size//2]) + 1
                
                freq_cycles_per_pixel = peak_index / block_size
                wavelength = 1 / freq_cycles_per_pixel if freq_cycles_per_pixel > 0 else 0
                
                # Filter out unrealistic wavelengths
                min_wave_length = 4
                max_wave_length = 20
                if min_wave_length < wavelength < max_wave_length:
                    frequencies.append(freq_cycles_per_pixel)

        if not frequencies:
            return default_return

        # Calculate standard deviation and convert to a 0-1 consistency score
        freq_std = np.std(frequencies)
        consistency_score = np.exp(-k * freq_std)

        return {"ridge_consistency": float(consistency_score)}

    except Exception:
        logger.exception("analyze_ridge_frequency failed")
        return default_return