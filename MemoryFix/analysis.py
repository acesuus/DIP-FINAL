"""
analysis.py - Image analysis functions for MemoryFix application.
Provides statistical calculations and histogram generation.
"""

import numpy as np
import cv2
from scipy import stats


def calculate_mean(image):
    """
    Calculate the mean pixel intensity of an image.
    
    Args:
        image: NumPy array (grayscale or color)
    
    Returns:
        Float mean value
    """
    if image is None:
        return 0.0
    return float(np.mean(image))


def calculate_std(image):
    """
    Calculate the standard deviation of pixel intensities.
    
    Args:
        image: NumPy array (grayscale or color)
    
    Returns:
        Float standard deviation value
    """
    if image is None:
        return 0.0
    return float(np.std(image))


def calculate_correlation(original, processed):
    """
    Calculate the Pearson correlation coefficient between original and processed images.
    Both images are converted to grayscale and resized to match if necessary.
    
    Args:
        original: Original image (NumPy array)
        processed: Processed image (NumPy array)
    
    Returns:
        Float correlation coefficient (-1 to 1)
    """
    if original is None or processed is None:
        return 0.0
    
    # Convert to grayscale if needed
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original.copy()
    
    if len(processed.shape) == 3:
        proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        proc_gray = processed.copy()
    
    # Resize processed to match original if different sizes
    if orig_gray.shape != proc_gray.shape:
        proc_gray = cv2.resize(proc_gray, (orig_gray.shape[1], orig_gray.shape[0]))
    
    # Flatten arrays for correlation calculation
    orig_flat = orig_gray.flatten().astype(np.float64)
    proc_flat = proc_gray.flatten().astype(np.float64)
    
    # Calculate Pearson correlation
    correlation, _ = stats.pearsonr(orig_flat, proc_flat)
    
    return float(correlation)


def generate_histogram_data(image):
    """
    Generate histogram data for an image.
    
    Args:
        image: NumPy array (grayscale or color)
    
    Returns:
        List of histogram arrays (one per channel)
    """
    if image is None:
        return []
    
    histograms = []
    
    if len(image.shape) == 2:
        # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        histograms.append(hist.flatten())
    else:
        # Color image - calculate for each channel
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            histograms.append(hist.flatten())
    
    return histograms


def get_full_analysis(original, processed):
    """
    Perform complete image analysis and return all statistics.
    
    Args:
        original: Original image (NumPy array)
        processed: Processed image (NumPy array)
    
    Returns:
        Dictionary with all analysis results
    """
    results = {
        "original_mean": calculate_mean(original),
        "original_std": calculate_std(original),
        "processed_mean": calculate_mean(processed),
        "processed_std": calculate_std(processed),
        "correlation": calculate_correlation(original, processed),
        "original_histogram": generate_histogram_data(original),
        "processed_histogram": generate_histogram_data(processed)
    }
    
    return results
