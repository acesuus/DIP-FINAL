"""
utils.py - Helper functions for MemoryFix application.
Handles image format conversions, validation, and display utilities.
"""

import os
import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


def cv_image_to_qpixmap(image, target_width=None, target_height=None):
    """
    Convert an OpenCV image (NumPy array) to QPixmap for PyQt6 display.
    Handles both grayscale and color (BGR) images.
    
    Args:
        image: NumPy array (grayscale or BGR color image)
        target_width: Optional width to scale the image
        target_height: Optional height to scale the image
    
    Returns:
        QPixmap object ready for display
    """
    if image is None:
        return QPixmap()
    
    # Make a copy to avoid modifying the original
    img = image.copy()
    
    # Ensure the image is uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Handle grayscale images
    if len(img.shape) == 2:
        height, width = img.shape
        bytes_per_line = width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
    # Handle color images (BGR from OpenCV)
    elif len(img.shape) == 3:
        # Convert BGR to RGB
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape
        bytes_per_line = channels * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    else:
        return QPixmap()
    
    pixmap = QPixmap.fromImage(q_image)
    
    # Scale if target dimensions are provided
    if target_width and target_height:
        pixmap = pixmap.scaled(
            target_width, target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    
    return pixmap


def resize_image_for_display(image, max_width=500, max_height=400):
    """
    Resize an image to fit within display bounds while maintaining aspect ratio.
    
    Args:
        image: NumPy array
        max_width: Maximum display width
        max_height: Maximum display height
    
    Returns:
        Resized NumPy array
    """
    import cv2
    
    if image is None:
        return None
    
    if len(image.shape) == 2:
        h, w = image.shape
    else:
        h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image


def validate_file_type(file_path):
    """
    Check if the given file path has a supported image extension.
    
    Args:
        file_path: Path to the file
    
    Returns:
        True if the file type is supported, False otherwise
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in supported_extensions


def get_output_path(original_path, output_dir="outputs", suffix="_restored"):
    """
    Generate an output file path that won't overwrite the original.
    
    Args:
        original_path: Original image file path
        output_dir: Directory for output files
        suffix: Suffix to add before extension
    
    Returns:
        New file path string
    """
    filename = os.path.basename(original_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}{suffix}{ext}"
    return os.path.join(output_dir, new_filename)


def get_image_info(image):
    """
    Get basic information about an image.
    
    Args:
        image: NumPy array
    
    Returns:
        Dictionary with width, height, channels, and dtype
    """
    if image is None:
        return {"width": 0, "height": 0, "channels": 0, "dtype": "None"}
    
    if len(image.shape) == 2:
        h, w = image.shape
        channels = 1
    else:
        h, w, channels = image.shape
    
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "dtype": str(image.dtype)
    }
