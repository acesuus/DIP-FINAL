"""
gui.py - Main PyQt6 GUI for MemoryFix application.
Contains the main window layout, buttons, image panels, sliders, and event handlers.
"""

import os
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QSlider, QComboBox,
    QGroupBox, QFrame, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QFont

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utils import cv_image_to_qpixmap, validate_file_type, get_image_info
from image_processor import (
    load_image, convert_to_grayscale, normalize_intensity,
    contrast_stretch, histogram_equalization, clahe_enhancement,
    median_filter, gaussian_filter, bilateral_filter,
    sharpen_image, adjust_brightness_contrast, restore_photo, save_image
)
from analysis import (
    calculate_mean, calculate_std, calculate_correlation,
    generate_histogram_data, get_full_analysis
)


# ============================================================
# STYLESHEET
# ============================================================

STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
}
QWidget {
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 10px;
    padding: 10px;
    padding-top: 25px;
    font-weight: bold;
    font-size: 13px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: #89b4fa;
}
QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: bold;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #585b70;
    border-color: #89b4fa;
}
QPushButton:pressed {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QPushButton#restoreBtn {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-size: 13px;
    padding: 10px;
}
QPushButton#restoreBtn:hover {
    background-color: #b4d0fb;
}
QPushButton#resetBtn {
    background-color: #f38ba8;
    color: #1e1e2e;
}
QPushButton#resetBtn:hover {
    background-color: #f5a0b8;
}
QLabel#imageLabel {
    background-color: #181825;
    border: 2px dashed #45475a;
    border-radius: 8px;
    color: #6c7086;
    font-size: 14px;
}
QLabel#statusLabel {
    color: #a6e3a1;
    font-size: 12px;
    padding: 4px;
}
QLabel#analysisLabel {
    color: #cdd6f4;
    font-size: 12px;
}
QSlider::groove:horizontal {
    background: #45475a;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #89b4fa;
    border-radius: 3px;
}
QComboBox {
    background-color: #45475a;
    border: 1px solid #585b70;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 22px;
}
QComboBox::drop-down {
    border: none;
}
QFrame#separator {
    background-color: #45475a;
    max-height: 1px;
}
"""


class MemoryFixWindow(QMainWindow):
    """Main application window for MemoryFix."""
    
    def __init__(self):
        super().__init__()
        
        # Image state
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        
        # Setup window
        self.setWindowTitle("MemoryFix - Photo Restoration Tool")
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(STYLESHEET)
        
        # Build the UI
        self._build_ui()
    
    def _build_ui(self):
        """Build the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Left sidebar
        sidebar = self._build_sidebar()
        main_layout.addWidget(sidebar)
        
        # Right content area (image panels + bottom analysis)
        content_layout = QVBoxLayout()
        content_layout.setSpacing(10)
        
        # Image panels (original + restored)
        image_panels = self._build_image_panels()
        content_layout.addWidget(image_panels, stretch=3)
        
        # Bottom analysis panel
        analysis_panel = self._build_analysis_panel()
        content_layout.addWidget(analysis_panel, stretch=1)
        
        main_layout.addLayout(content_layout, stretch=1)
    
    def _build_sidebar(self):
        """Build the left sidebar with buttons and controls."""
        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(230)
        sidebar_widget.setStyleSheet("background-color: #11111b; border-radius: 10px;")
        
        layout = QVBoxLayout(sidebar_widget)
        layout.setContentsMargins(12, 15, 12, 15)
        layout.setSpacing(6)
        
        # Title
        title = QLabel("MemoryFix")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #89b4fa; background: transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Photo Restoration")
        subtitle.setStyleSheet("color: #6c7086; font-size: 11px; background: transparent;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addSpacing(10)
        
        # Action buttons
        btn_data = [
            ("Import Image", self._on_import),
            ("Preprocess", self._on_preprocess),
            ("Enhance Contrast", self._on_enhance_contrast),
            ("Reduce Noise", self._on_reduce_noise),
            ("Sharpen Image", self._on_sharpen),
        ]
        
        for text, handler in btn_data:
            btn = QPushButton(text)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(handler)
            layout.addWidget(btn)
        
        # Restore button (highlighted)
        restore_btn = QPushButton("Restore Photo")
        restore_btn.setObjectName("restoreBtn")
        restore_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        restore_btn.clicked.connect(self._on_restore)
        layout.addWidget(restore_btn)
        
        # Separator
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)
        layout.addSpacing(4)
        
        # Analysis and save buttons
        btn_data2 = [
            ("Analyze Image", self._on_analyze),
            ("Save Output", self._on_save),
        ]
        for text, handler in btn_data2:
            btn = QPushButton(text)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(handler)
            layout.addWidget(btn)
        
        # Reset button (red)
        reset_btn = QPushButton("Reset")
        reset_btn.setObjectName("resetBtn")
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(reset_btn)
        
        layout.addSpacing(8)
        
        # Separator
        sep2 = QFrame()
        sep2.setObjectName("separator")
        sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)
        layout.addSpacing(4)
        
        # Controls section
        controls_label = QLabel("Adjustments")
        controls_label.setStyleSheet("color: #89b4fa; font-weight: bold; font-size: 12px; background: transparent;")
        layout.addWidget(controls_label)
        
        # Brightness slider
        layout.addWidget(self._make_slider_label("Brightness"))
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setStyleSheet("background: transparent;")
        layout.addWidget(self.brightness_slider)
        
        # Contrast slider
        layout.addWidget(self._make_slider_label("Contrast"))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(50, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setStyleSheet("background: transparent;")
        layout.addWidget(self.contrast_slider)
        
        # Sharpness slider
        layout.addWidget(self._make_slider_label("Sharpness"))
        self.sharpness_slider = QSlider(Qt.Orientation.Horizontal)
        self.sharpness_slider.setRange(0, 200)
        self.sharpness_slider.setValue(100)
        self.sharpness_slider.setStyleSheet("background: transparent;")
        layout.addWidget(self.sharpness_slider)
        
        # Filter type dropdown
        layout.addWidget(self._make_slider_label("Filter Type"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Median", "Gaussian", "Bilateral"])
        self.filter_combo.setStyleSheet("background-color: #45475a;")
        layout.addWidget(self.filter_combo)
        
        # Filter strength slider
        layout.addWidget(self._make_slider_label("Filter Strength"))
        self.filter_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.filter_strength_slider.setRange(1, 15)
        self.filter_strength_slider.setValue(3)
        self.filter_strength_slider.setSingleStep(2)
        self.filter_strength_slider.setStyleSheet("background: transparent;")
        layout.addWidget(self.filter_strength_slider)
        
        # Apply adjustments button
        apply_btn = QPushButton("Apply Adjustments")
        apply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        apply_btn.clicked.connect(self._on_apply_adjustments)
        layout.addWidget(apply_btn)
        
        layout.addStretch()
        
        return sidebar_widget
    
    def _make_slider_label(self, text):
        """Create a small label for sliders."""
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #a6adc8; font-size: 11px; background: transparent; margin-top: 4px;")
        return lbl
    
    def _build_image_panels(self):
        """Build the original and restored image preview panels."""
        panel_widget = QWidget()
        layout = QHBoxLayout(panel_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Original image panel
        orig_group = QGroupBox("Original Image")
        orig_layout = QVBoxLayout(orig_group)
        self.original_label = QLabel("No image loaded")
        self.original_label.setObjectName("imageLabel")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(350, 300)
        self.original_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        orig_layout.addWidget(self.original_label)
        layout.addWidget(orig_group)
        
        # Restored image panel
        rest_group = QGroupBox("Restored Image")
        rest_layout = QVBoxLayout(rest_group)
        self.restored_label = QLabel("No processed image")
        self.restored_label.setObjectName("imageLabel")
        self.restored_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.restored_label.setMinimumSize(350, 300)
        self.restored_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rest_layout.addWidget(self.restored_label)
        layout.addWidget(rest_group)
        
        return panel_widget
    
    def _build_analysis_panel(self):
        """Build the bottom analysis and status panel."""
        panel = QGroupBox("Image Analysis")
        layout = QHBoxLayout(panel)
        layout.setSpacing(20)
        
        # Stats section
        stats_layout = QVBoxLayout()
        
        self.size_label = QLabel("Size: --")
        self.size_label.setObjectName("analysisLabel")
        stats_layout.addWidget(self.size_label)
        
        self.mean_label = QLabel("Mean: --")
        self.mean_label.setObjectName("analysisLabel")
        stats_layout.addWidget(self.mean_label)
        
        self.std_label = QLabel("Std Dev: --")
        self.std_label.setObjectName("analysisLabel")
        stats_layout.addWidget(self.std_label)
        
        self.corr_label = QLabel("Correlation: --")
        self.corr_label.setObjectName("analysisLabel")
        stats_layout.addWidget(self.corr_label)
        
        layout.addLayout(stats_layout)
        
        # Histogram canvas
        self.hist_figure = Figure(figsize=(5, 2), dpi=80)
        self.hist_figure.patch.set_facecolor('#313244')
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_canvas.setMinimumHeight(120)
        self.hist_canvas.setStyleSheet("background-color: #313244; border-radius: 4px;")
        layout.addWidget(self.hist_canvas, stretch=1)
        
        # Status section
        status_layout = QVBoxLayout()
        status_title = QLabel("Status")
        status_title.setStyleSheet("color: #89b4fa; font-weight: bold; font-size: 12px;")
        status_layout.addWidget(status_title)
        
        self.status_label = QLabel("Ready. Import an image to begin.")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        layout.addLayout(status_layout)
        
        return panel
    
    # ============================================================
    # IMAGE DISPLAY HELPERS
    # ============================================================
    
    def _display_original(self):
        """Display the original image in the left panel."""
        if self.original_image is not None:
            pixmap = cv_image_to_qpixmap(self.original_image)
            scaled = pixmap.scaled(
                self.original_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.original_label.setPixmap(scaled)
    
    def _display_processed(self):
        """Display the processed image in the right panel."""
        if self.processed_image is not None:
            pixmap = cv_image_to_qpixmap(self.processed_image)
            scaled = pixmap.scaled(
                self.restored_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.restored_label.setPixmap(scaled)
    
    def _update_status(self, message):
        """Update the status label."""
        self.status_label.setText(message)
    
    def _check_image_loaded(self):
        """Check if an image is loaded, show warning if not."""
        if self.original_image is None:
            QMessageBox.warning(
                self, "No Image",
                "Please import an image first."
            )
            return False
        return True
    
    def _check_processed_exists(self):
        """Check if a processed image exists."""
        if self.processed_image is None:
            QMessageBox.warning(
                self, "No Processed Image",
                "Please process an image first before performing this action."
            )
            return False
        return True
    
    # ============================================================
    # BUTTON EVENT HANDLERS
    # ============================================================
    
    def _on_import(self):
        """Handle Import Image button."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif);;All Files (*)"
        )
        
        if not file_path:
            return
        
        if not validate_file_type(file_path):
            QMessageBox.warning(self, "Invalid File", "Unsupported file format.")
            return
        
        # Load image
        image = load_image(file_path)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load the image.")
            return
        
        self.original_image = image
        self.processed_image = None
        self.image_path = file_path
        
        # Display
        self._display_original()
        self.restored_label.clear()
        self.restored_label.setText("No processed image")
        
        # Update info
        info = get_image_info(image)
        self.size_label.setText(f"Size: {info['width']} x {info['height']} ({info['channels']}ch)")
        self._clear_analysis()
        self._update_status("Image loaded successfully.")
    
    def _on_preprocess(self):
        """Handle Preprocess button."""
        if not self._check_image_loaded():
            return
        
        # Convert to grayscale and normalize
        gray = convert_to_grayscale(self.original_image)
        normalized = normalize_intensity(gray)
        
        self.processed_image = normalized
        self._display_processed()
        self._update_status("Preprocessing complete. Image converted to grayscale and normalized.")
    
    def _on_enhance_contrast(self):
        """Handle Enhance Contrast button."""
        if not self._check_image_loaded():
            return
        
        # Apply CLAHE to the current working image
        source = self.processed_image if self.processed_image is not None else self.original_image
        enhanced = clahe_enhancement(source)
        
        self.processed_image = enhanced
        self._display_processed()
        self._update_status("Contrast enhancement (CLAHE) applied.")
    
    def _on_reduce_noise(self):
        """Handle Reduce Noise button."""
        if not self._check_image_loaded():
            return
        
        source = self.processed_image if self.processed_image is not None else self.original_image
        
        # Get filter type from combo box
        filter_type = self.filter_combo.currentText()
        strength = self.filter_strength_slider.value()
        
        # Ensure odd kernel size
        kernel_size = strength if strength % 2 == 1 else strength + 1
        
        if filter_type == "Median":
            result = median_filter(source, kernel_size=kernel_size)
        elif filter_type == "Gaussian":
            result = gaussian_filter(source, kernel_size=kernel_size)
        else:  # Bilateral
            result = bilateral_filter(source)
        
        self.processed_image = result
        self._display_processed()
        self._update_status(f"Noise reduction ({filter_type} filter) applied.")
    
    def _on_sharpen(self):
        """Handle Sharpen Image button."""
        if not self._check_image_loaded():
            return
        
        source = self.processed_image if self.processed_image is not None else self.original_image
        
        # Get sharpness value from slider (100 = 1.0x)
        strength = self.sharpness_slider.value() / 100.0
        
        sharpened = sharpen_image(source, strength=strength)
        
        self.processed_image = sharpened
        self._display_processed()
        self._update_status("Image sharpening applied.")
    
    def _on_restore(self):
        """Handle Restore Photo button - applies full pipeline."""
        if not self._check_image_loaded():
            return
        
        restored = restore_photo(self.original_image)
        
        self.processed_image = restored
        self._display_processed()
        self._update_status("Photo restoration completed. Full pipeline applied.")
    
    def _on_analyze(self):
        """Handle Analyze Image button."""
        if not self._check_image_loaded():
            return
        if not self._check_processed_exists():
            return
        
        # Calculate statistics
        analysis = get_full_analysis(self.original_image, self.processed_image)
        
        # Update labels
        self.mean_label.setText(f"Mean: {analysis['processed_mean']:.2f}")
        self.std_label.setText(f"Std Dev: {analysis['processed_std']:.2f}")
        self.corr_label.setText(f"Correlation: {analysis['correlation']:.4f}")
        
        # Draw histograms
        self._draw_histograms(analysis)
        
        self._update_status("Image analysis completed.")
    
    def _on_save(self):
        """Handle Save Output button."""
        if not self._check_processed_exists():
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Restored Image", "restored_image.png",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if not file_path:
            return
        
        success = save_image(self.processed_image, file_path)
        
        if success:
            self._update_status(f"Image saved successfully to: {os.path.basename(file_path)}")
        else:
            QMessageBox.warning(self, "Error", "Failed to save the image.")
    
    def _on_reset(self):
        """Handle Reset button."""
        self.processed_image = None
        
        # Clear restored panel
        self.restored_label.clear()
        self.restored_label.setText("No processed image")
        
        # Reset sliders
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.sharpness_slider.setValue(100)
        self.filter_strength_slider.setValue(3)
        self.filter_combo.setCurrentIndex(0)
        
        # Clear analysis
        self._clear_analysis()
        
        # Clear histogram
        self.hist_figure.clear()
        self.hist_canvas.draw()
        
        self._update_status("Reset complete. Original image preserved.")
    
    def _on_apply_adjustments(self):
        """Handle Apply Adjustments button."""
        if not self._check_image_loaded():
            return
        
        source = self.processed_image if self.processed_image is not None else self.original_image
        
        # Get slider values
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value() / 100.0
        sharpness = self.sharpness_slider.value() / 100.0
        
        # Apply brightness and contrast
        result = adjust_brightness_contrast(source, brightness=brightness, contrast=contrast)
        
        # Apply sharpening if sharpness != 1.0
        if abs(sharpness - 1.0) > 0.05:
            result = sharpen_image(result, strength=sharpness)
        
        self.processed_image = result
        self._display_processed()
        self._update_status(f"Adjustments applied (B:{brightness}, C:{contrast:.2f}, S:{sharpness:.2f}).")
    
    # ============================================================
    # ANALYSIS HELPERS
    # ============================================================
    
    def _clear_analysis(self):
        """Clear analysis labels."""
        self.mean_label.setText("Mean: --")
        self.std_label.setText("Std Dev: --")
        self.corr_label.setText("Correlation: --")
    
    def _draw_histograms(self, analysis):
        """Draw histogram comparison on the embedded Matplotlib canvas."""
        self.hist_figure.clear()
        
        ax1 = self.hist_figure.add_subplot(121)
        ax2 = self.hist_figure.add_subplot(122)
        
        # Style axes
        for ax in [ax1, ax2]:
            ax.set_facecolor('#181825')
            ax.tick_params(colors='#cdd6f4', labelsize=7)
            ax.spines['bottom'].set_color('#45475a')
            ax.spines['left'].set_color('#45475a')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Original histogram
        orig_hist = analysis['original_histogram']
        if orig_hist:
            if len(orig_hist) == 1:
                ax1.plot(orig_hist[0], color='#89b4fa', linewidth=0.8)
                ax1.fill_between(range(256), orig_hist[0], alpha=0.3, color='#89b4fa')
            else:
                colors = ['#89b4fa', '#a6e3a1', '#f38ba8']
                for i, hist in enumerate(orig_hist):
                    ax1.plot(hist, color=colors[i], linewidth=0.7, alpha=0.8)
        ax1.set_title("Original", color='#cdd6f4', fontsize=9)
        ax1.set_xlim([0, 256])
        
        # Processed histogram
        proc_hist = analysis['processed_histogram']
        if proc_hist:
            if len(proc_hist) == 1:
                ax2.plot(proc_hist[0], color='#a6e3a1', linewidth=0.8)
                ax2.fill_between(range(256), proc_hist[0], alpha=0.3, color='#a6e3a1')
            else:
                colors = ['#89b4fa', '#a6e3a1', '#f38ba8']
                for i, hist in enumerate(proc_hist):
                    ax2.plot(hist, color=colors[i], linewidth=0.7, alpha=0.8)
        ax2.set_title("Processed", color='#cdd6f4', fontsize=9)
        ax2.set_xlim([0, 256])
        
        self.hist_figure.tight_layout(pad=1.0)
        self.hist_canvas.draw()
    
    def resizeEvent(self, event):
        """Handle window resize to update image display."""
        super().resizeEvent(event)
        self._display_original()
        self._display_processed()
