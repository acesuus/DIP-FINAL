"""
image_processor.py - Core image processing functions for MemoryFix application.
Implements preprocessing, contrast enhancement, noise reduction, sharpening,
and the main photo restoration pipeline.
"""

import cv2
import numpy as np


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def load_image(file_path):
    """
    Load an image from file using OpenCV.
    
    Args:
        file_path: Path to the image file
    
    Returns:
        NumPy array (BGR color image) or None if loading fails
    """
    image = cv2.imread(file_path)
    return image


def convert_to_grayscale(image):
    """
    Convert a color image to grayscale.
    
    Args:
        image: NumPy array (BGR or already grayscale)
    
    Returns:
        Grayscale NumPy array
    """
    if image is None:
        return None
    
    if len(image.shape) == 2:
        return image.copy()
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_intensity(image):
    """
    Normalize image intensity to use full 0-255 range.
    
    Args:
        image: NumPy array
    
    Returns:
        Normalized NumPy array
    """
    if image is None:
        return None
    
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


# ============================================================
# CONTRAST ENHANCEMENT FUNCTIONS
# ============================================================

def contrast_stretch(image):
    """
    Apply contrast stretching (min-max normalization).
    Stretches the intensity range to cover 0-255.
    
    Args:
        image: NumPy array (grayscale or color)
    
    Returns:
        Contrast-stretched NumPy array
    """
    if image is None:
        return None
    
    if len(image.shape) == 2:
        # Grayscale
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val == 0:
            return image.copy()
        stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return stretched
    else:
        # Color - apply to each channel
        result = np.zeros_like(image)
        for i in range(3):
            channel = image[:, :, i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val - min_val == 0:
                result[:, :, i] = channel
            else:
                result[:, :, i] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return result


def histogram_equalization(image):
    """
    Apply histogram equalization to improve contrast.
    
    Args:
        image: NumPy array (grayscale or color)
    
    Returns:
        Equalized NumPy array
    """
    if image is None:
        return None
    
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    else:
        # Convert to YCrCb and equalize the Y channel
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Provides better local contrast enhancement without over-amplifying noise.
    
    Args:
        image: NumPy array (grayscale or color)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        CLAHE-enhanced NumPy array
    """
    if image is None:
        return None
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if len(image.shape) == 2:
        return clahe.apply(image)
    else:
        # Convert to LAB color space and apply CLAHE to L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================
# NOISE REDUCTION FUNCTIONS
# ============================================================

def median_filter(image, kernel_size=3):
    """
    Apply median filtering to reduce salt-and-pepper noise.
    
    Args:
        image: NumPy array
        kernel_size: Size of the median filter kernel (must be odd)
    
    Returns:
        Filtered NumPy array
    """
    if image is None:
        return None
    
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.medianBlur(image, kernel_size)


def gaussian_filter(image, kernel_size=5, sigma=0):
    """
    Apply Gaussian filtering to reduce noise.
    
    Args:
        image: NumPy array
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation (0 means auto-calculated)
    
    Returns:
        Filtered NumPy array
    """
    if image is None:
        return None
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filtering to reduce noise while keeping edges sharp.
    
    Args:
        image: NumPy array
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
    
    Returns:
        Filtered NumPy array
    """
    if image is None:
        return None
    
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# ============================================================
# SHARPENING FUNCTIONS
# ============================================================

def sharpen_image(image, strength=1.0):
    """
    Sharpen an image using kernel convolution.
    
    Args:
        image: NumPy array
        strength: Sharpening strength multiplier (1.0 = normal)
    
    Returns:
        Sharpened NumPy array
    """
    if image is None:
        return None
    
    # Sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    # Adjust kernel strength
    if strength != 1.0:
        # Blend between identity and sharpening kernel
        identity = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        kernel = identity + strength * (kernel - identity)
    
    sharpened = cv2.filter2D(image, -1, kernel)
    
    # Clip values to valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


# ============================================================
# BRIGHTNESS AND CONTRAST ADJUSTMENT
# ============================================================

def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """
    Adjust brightness and contrast of an image.
    
    Formula: output = contrast * input + brightness
    
    Args:
        image: NumPy array
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast multiplier (0.5 to 2.0)
    
    Returns:
        Adjusted NumPy array
    """
    if image is None:
        return None
    
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted


# ============================================================
# MAIN RESTORATION PIPELINE
# ============================================================

def restore_photo(image):
    """
    Apply the complete photo restoration pipeline.
    
    Pipeline:
    1. Denoising (bilateral filter for edge-preserving smoothing)
    2. Contrast enhancement (CLAHE)
    3. Sharpening (kernel convolution)
    4. Slight brightness and contrast adjustment
    
    Args:
        image: NumPy array (original image)
    
    Returns:
        Restored NumPy array
    """
    if image is None:
        return None
    
    # Step 1: Denoising - bilateral filter preserves edges
    denoised = bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)
    
    # Step 2: Contrast enhancement using CLAHE
    enhanced = clahe_enhancement(denoised, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Step 3: Sharpening
    sharpened = sharpen_image(enhanced, strength=0.8)
    
    # Step 4: Slight brightness and contrast adjustment
    restored = adjust_brightness_contrast(sharpened, brightness=5, contrast=1.1)
    
    return restored


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_image(image, file_path):
    """
    Save an image to file.
    
    Args:
        image: NumPy array to save
        file_path: Output file path
    
    Returns:
        True if successful, False otherwise
    """
    if image is None:
        return False
    
    try:
        cv2.imwrite(file_path, image)
        return True
    except Exception:
        return False
