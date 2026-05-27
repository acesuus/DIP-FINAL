"""
gui.py - Main PyQt6 GUI for MemoryFix application.

The interface has two modes that can be switched at the top of the window:

    Simple Mode (default)
        - Beginner-friendly: Upload Photo, Enhance Photo, Save Result, Reset.
        - "Enhance Photo" runs the full restore_photo() pipeline behind the scenes.
        - Optional "View Analysis" button reveals statistics in beginner-friendly
          labels (Mean Brightness, Contrast Level, Similarity to Original).

    Advanced Mode
        - Step-by-step Digital Image Processing simulator for student demos.
        - Runs each technique individually so the effect of each step is visible:
              1. Grayscale
              2. Normalize
              3. Contrast Enhancement (CLAHE)
              4. Noise Reduction (Median filter)
              5. Sharpening (kernel convolution)
              6. Final Restoration (brightness/contrast polish)
              7. Image Analysis
              8. Histogram Comparison
        - Controls: Run Current Step, Run Next Step, Run All Steps, Reset.
"""

import os

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QGroupBox, QFrame,
    QSizePolicy, QStackedWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utils import cv_image_to_qpixmap, validate_file_type, get_image_info
from image_processor import (
    load_image,
    convert_to_grayscale,
    normalize_intensity,
    clahe_enhancement,
    median_filter,
    sharpen_image,
    adjust_brightness_contrast,
    restore_photo,
    save_image,
)
from analysis import get_full_analysis


# ============================================================
# STYLESHEET
# ============================================================

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

QGroupBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px;
    padding-top: 26px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: #89b4fa;
}

/* ---------- Header / mode toggle ---------- */
QWidget#headerBar {
    background-color: #181825;
    border-radius: 10px;
}
QLabel#appTitle {
    color: #89b4fa;
    background: transparent;
}
QLabel#appSubtitle {
    color: #6c7086;
    background: transparent;
    font-size: 11px;
}
QWidget#modeToggle {
    background-color: #313244;
    border-radius: 10px;
    padding: 4px;
}
QPushButton#modeBtn {
    background-color: transparent;
    border: none;
    color: #a6adc8;
    font-weight: bold;
    padding: 8px 22px;
    border-radius: 7px;
    min-width: 110px;
}
QPushButton#modeBtn:checked {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QPushButton#modeBtn:hover:!checked {
    color: #cdd6f4;
}

/* ---------- Image panels ---------- */
QLabel#imageLabel {
    background-color: #181825;
    border: 2px dashed #45475a;
    border-radius: 8px;
    color: #6c7086;
    font-size: 14px;
}

/* ---------- Buttons ---------- */
QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #585b70;
    border-color: #89b4fa;
}
QPushButton:pressed {
    background-color: #89b4fa;
    color: #1e1e2e;
}

/* Big simple-mode buttons */
QPushButton#bigBtn {
    border-radius: 10px;
    padding: 16px 18px;
    font-size: 14px;
    min-height: 24px;
}
QPushButton#primaryBtn {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
}
QPushButton#primaryBtn:hover {
    background-color: #b4d0fb;
}
QPushButton#accentBtn {
    background-color: #a6e3a1;
    color: #1e1e2e;
    border: none;
    font-size: 15px;
}
QPushButton#accentBtn:hover {
    background-color: #c1f0bd;
}
QPushButton#dangerBtn {
    background-color: #f38ba8;
    color: #1e1e2e;
    border: none;
}
QPushButton#dangerBtn:hover {
    background-color: #f5a0b8;
}

/* ---------- Status / labels ---------- */
QLabel#statusLabel {
    color: #a6e3a1;
    font-size: 13px;
    padding: 8px 4px;
}
QLabel#analysisLabel {
    color: #cdd6f4;
    font-size: 13px;
    padding: 2px 0;
}
QLabel#analysisValue {
    color: #f9e2af;
    font-weight: bold;
    font-size: 13px;
}
QLabel#currentStepLabel {
    color: #f9e2af;
    background-color: #181825;
    border-radius: 6px;
    padding: 8px;
    font-weight: bold;
}

/* Step list items */
QLabel#stepItem {
    color: #6c7086;
    background-color: transparent;
    padding: 6px 10px;
    border-radius: 4px;
}
QLabel#stepItemActive {
    color: #1e1e2e;
    background-color: #f9e2af;
    padding: 6px 10px;
    border-radius: 4px;
    font-weight: bold;
}
QLabel#stepItemDone {
    color: #a6e3a1;
    background-color: transparent;
    padding: 6px 10px;
    border-radius: 4px;
}

QFrame#sep {
    background-color: #45475a;
    max-height: 1px;
    margin: 4px 0;
}
"""


# ============================================================
# REUSABLE IMAGE PANEL
# ============================================================

class ImagePanel(QGroupBox):
    """A titled panel that displays an image and rescales it on resize
    while preserving aspect ratio."""

    def __init__(self, title: str, placeholder: str = "No image"):
        super().__init__(title)
        self._image = None
        self._placeholder = placeholder

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 22, 10, 10)

        self.label = QLabel(placeholder)
        self.label.setObjectName("imageLabel")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(360, 280)
        self.label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.label)

    def set_image(self, image):
        self._image = image
        self._refresh()

    def clear_image(self):
        self._image = None
        self.label.clear()
        self.label.setText(self._placeholder)

    def set_title(self, title: str):
        self.setTitle(title)

    def _refresh(self):
        if self._image is None:
            return
        pixmap = cv_image_to_qpixmap(self._image)
        scaled = pixmap.scaled(
            self.label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


# ============================================================
# MAIN WINDOW
# ============================================================

# The Advanced Mode pipeline. Each entry is (display_name, internal_id).
# These ids are used to dispatch processing in _execute_step().
ADVANCED_STEPS = [
    ("Grayscale", "grayscale"),
    ("Normalize Intensity", "normalize"),
    ("Contrast Enhancement", "contrast"),
    ("Noise Reduction", "denoise"),
    ("Sharpening", "sharpen"),
    ("Final Restoration", "final"),
    ("Image Analysis", "analyze"),
    ("Histogram Comparison", "histogram"),
]


class MemoryFixWindow(QMainWindow):
    """Main application window with Simple and Advanced modes."""

    def __init__(self):
        super().__init__()

        # ---- Shared state ----
        self.original_image = None
        self.processed_image = None
        self.image_path = None

        # Advanced-mode state
        self.current_step_index = 0
        self.last_analysis = None  # cached analysis dict for histogram redraws

        # ---- Window setup ----
        self.setWindowTitle("MemoryFix - Photo Restoration Tool")
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(STYLESHEET)

        self._build_ui()
        self._switch_mode(0)  # Simple mode by default

    # --------------------------------------------------------
    # UI construction
    # --------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        root.addWidget(self._build_header())

        self.stack = QStackedWidget()
        self.simple_view = self._build_simple_view()
        self.advanced_view = self._build_advanced_view()
        self.stack.addWidget(self.simple_view)
        self.stack.addWidget(self.advanced_view)
        root.addWidget(self.stack, stretch=1)

    def _build_header(self):
        bar = QWidget()
        bar.setObjectName("headerBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 10, 16, 10)

        # Title block
        title_block = QVBoxLayout()
        title_block.setSpacing(0)
        title = QLabel("MemoryFix")
        title.setObjectName("appTitle")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        subtitle = QLabel("Offline Old Photo Restoration")
        subtitle.setObjectName("appSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        layout.addLayout(title_block)

        layout.addStretch()

        # Mode toggle (segmented control)
        toggle = QWidget()
        toggle.setObjectName("modeToggle")
        toggle_layout = QHBoxLayout(toggle)
        toggle_layout.setContentsMargins(4, 4, 4, 4)
        toggle_layout.setSpacing(4)

        self.simple_btn = QPushButton("Simple Mode")
        self.simple_btn.setObjectName("modeBtn")
        self.simple_btn.setCheckable(True)
        self.simple_btn.setChecked(True)
        self.simple_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.simple_btn.clicked.connect(lambda: self._switch_mode(0))

        self.advanced_btn = QPushButton("Advanced Mode")
        self.advanced_btn.setObjectName("modeBtn")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.advanced_btn.clicked.connect(lambda: self._switch_mode(1))

        toggle_layout.addWidget(self.simple_btn)
        toggle_layout.addWidget(self.advanced_btn)
        layout.addWidget(toggle)

        return bar

    # ---------------- Simple Mode ----------------
    def _build_simple_view(self):
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Image panels
        panels = QHBoxLayout()
        panels.setSpacing(12)
        self.simple_orig_panel = ImagePanel("Original Image", "Upload a photo to begin")
        self.simple_proc_panel = ImagePanel("Enhanced Image", "Click Enhance Photo")
        panels.addWidget(self.simple_orig_panel)
        panels.addWidget(self.simple_proc_panel)
        layout.addLayout(panels, stretch=3)

        # Big action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        upload = self._make_big_button("Upload Photo", "primaryBtn")
        upload.clicked.connect(self._on_upload)

        enhance = self._make_big_button("Enhance Photo", "accentBtn")
        enhance.clicked.connect(self._on_enhance_simple)

        view_analysis = self._make_big_button("View Analysis", "bigBtn")
        view_analysis.clicked.connect(self._on_view_analysis)

        save = self._make_big_button("Save Result", "primaryBtn")
        save.clicked.connect(self._on_save)

        reset = self._make_big_button("Reset", "dangerBtn")
        reset.clicked.connect(self._on_reset_simple)

        btn_row.addWidget(upload)
        btn_row.addWidget(enhance)
        btn_row.addWidget(view_analysis)
        btn_row.addWidget(save)
        btn_row.addWidget(reset)
        layout.addLayout(btn_row)

        # Optional analysis card (hidden until "View Analysis" is clicked)
        self.simple_analysis_box = QGroupBox("Image Analysis")
        self.simple_analysis_box.setVisible(False)
        sa_layout = QHBoxLayout(self.simple_analysis_box)
        sa_layout.setSpacing(30)

        self.simple_mean_value = self._make_stat_pair(
            sa_layout, "Mean Brightness"
        )
        self.simple_std_value = self._make_stat_pair(
            sa_layout, "Contrast Level"
        )
        self.simple_corr_value = self._make_stat_pair(
            sa_layout, "Similarity to Original"
        )
        sa_layout.addStretch()
        layout.addWidget(self.simple_analysis_box)

        # Status
        self.simple_status = QLabel("Upload an old photo to begin.")
        self.simple_status.setObjectName("statusLabel")
        layout.addWidget(self.simple_status)

        return view

    def _make_big_button(self, text: str, style_id: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("bigBtn" if style_id == "bigBtn" else style_id)
        # All big buttons share the bigBtn padding via property name fallback.
        if style_id != "bigBtn":
            # Apply both classes by combining QSS via dynamic property + setObjectName.
            # PyQt's QSS only matches one objectName, so we duplicate the size rules
            # in primaryBtn/accentBtn/dangerBtn implicitly via min-height padding
            # below.
            btn.setMinimumHeight(56)
            btn.setStyleSheet("font-size: 14px; font-weight: bold; border-radius: 10px;")
        else:
            btn.setMinimumHeight(56)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    def _make_stat_pair(self, parent_layout, label_text: str) -> QLabel:
        """Add a label + value pair to a horizontal layout, return the value QLabel."""
        block = QVBoxLayout()
        lbl = QLabel(label_text)
        lbl.setObjectName("analysisLabel")
        val = QLabel("--")
        val.setObjectName("analysisValue")
        block.addWidget(lbl)
        block.addWidget(val)
        parent_layout.addLayout(block)
        return val

    # ---------------- Advanced Mode ----------------
    def _build_advanced_view(self):
        view = QWidget()
        layout = QHBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ---- Sidebar (steps + controls) ----
        sidebar = QWidget()
        sidebar.setFixedWidth(290)
        sidebar.setStyleSheet("background-color: #181825; border-radius: 10px;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(12, 14, 12, 14)
        side_layout.setSpacing(8)

        upload_btn = QPushButton("Upload Photo")
        upload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        upload_btn.clicked.connect(self._on_upload)
        upload_btn.setStyleSheet(
            "background-color: #89b4fa; color: #1e1e2e; padding: 10px;"
            "border-radius: 6px; font-weight: bold;"
        )
        side_layout.addWidget(upload_btn)

        # Current step indicator
        self.current_step_label = QLabel("Current Step: -")
        self.current_step_label.setObjectName("currentStepLabel")
        self.current_step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_layout.addWidget(self.current_step_label)

        # Step list
        steps_box = QGroupBox("DIP Process Steps")
        steps_layout = QVBoxLayout(steps_box)
        steps_layout.setSpacing(2)
        self.step_item_labels = []
        for i, (name, _) in enumerate(ADVANCED_STEPS):
            lbl = QLabel(f"{i + 1}. {name}")
            lbl.setObjectName("stepItem")
            self.step_item_labels.append(lbl)
            steps_layout.addWidget(lbl)
        side_layout.addWidget(steps_box)

        # Step control buttons
        run_current = QPushButton("Run Current Step")
        run_current.setCursor(Qt.CursorShape.PointingHandCursor)
        run_current.clicked.connect(self._on_run_current)

        run_next = QPushButton("Run Next Step")
        run_next.setCursor(Qt.CursorShape.PointingHandCursor)
        run_next.clicked.connect(self._on_run_next)
        run_next.setStyleSheet(
            "background-color: #a6e3a1; color: #1e1e2e; padding: 10px;"
            "border-radius: 6px; font-weight: bold;"
        )

        run_all = QPushButton("Run All Steps")
        run_all.setCursor(Qt.CursorShape.PointingHandCursor)
        run_all.clicked.connect(self._on_run_all)

        reset_adv = QPushButton("Reset Advanced Process")
        reset_adv.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_adv.clicked.connect(self._on_reset_advanced)
        reset_adv.setStyleSheet(
            "background-color: #f38ba8; color: #1e1e2e; padding: 8px;"
            "border-radius: 6px; font-weight: bold;"
        )

        save_btn = QPushButton("Save Result")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self._on_save)

        side_layout.addWidget(run_current)
        side_layout.addWidget(run_next)
        side_layout.addWidget(run_all)

        sep = QFrame()
        sep.setObjectName("sep")
        sep.setFrameShape(QFrame.Shape.HLine)
        side_layout.addWidget(sep)

        side_layout.addWidget(reset_adv)
        side_layout.addWidget(save_btn)
        side_layout.addStretch()

        layout.addWidget(sidebar)

        # ---- Right side: image panels + analysis/histogram ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        panels = QHBoxLayout()
        panels.setSpacing(12)
        self.adv_orig_panel = ImagePanel("Original Image", "Upload a photo to begin")
        self.adv_proc_panel = ImagePanel(
            "Current Step Output", "Click Run Next Step"
        )
        panels.addWidget(self.adv_orig_panel)
        panels.addWidget(self.adv_proc_panel)
        right_layout.addLayout(panels, stretch=3)

        # Analysis + histogram
        bottom = QGroupBox("Analysis and Histogram")
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setSpacing(20)

        stats_block = QVBoxLayout()
        self.adv_size_value = self._make_simple_stat(stats_block, "Size")
        self.adv_mean_value = self._make_simple_stat(stats_block, "Mean Brightness")
        self.adv_std_value = self._make_simple_stat(stats_block, "Contrast Level")
        self.adv_corr_value = self._make_simple_stat(
            stats_block, "Similarity to Original"
        )
        stats_block.addStretch()
        bottom_layout.addLayout(stats_block)

        self.adv_hist_figure = Figure(figsize=(5, 2.2), dpi=80, facecolor="#313244")
        self.adv_hist_canvas = FigureCanvas(self.adv_hist_figure)
        self.adv_hist_canvas.setMinimumHeight(150)
        bottom_layout.addWidget(self.adv_hist_canvas, stretch=1)

        right_layout.addWidget(bottom, stretch=2)

        # Status
        self.adv_status = QLabel("Upload a photo, then run steps to see each effect.")
        self.adv_status.setObjectName("statusLabel")
        right_layout.addWidget(self.adv_status)

        layout.addWidget(right, stretch=1)

        # Initialize the step highlight
        self._refresh_step_highlights()

        return view

    def _make_simple_stat(self, parent_layout, label_text: str) -> QLabel:
        row = QHBoxLayout()
        lbl = QLabel(f"{label_text}:")
        lbl.setObjectName("analysisLabel")
        lbl.setMinimumWidth(160)
        val = QLabel("--")
        val.setObjectName("analysisValue")
        row.addWidget(lbl)
        row.addWidget(val)
        row.addStretch()
        parent_layout.addLayout(row)
        return val

    # --------------------------------------------------------
    # Mode switching
    # --------------------------------------------------------
    def _switch_mode(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self.simple_btn.setChecked(idx == 0)
        self.advanced_btn.setChecked(idx == 1)
        self._refresh_displays()

    def _refresh_displays(self):
        """Show the current original/processed images in whichever view is active."""
        if self.original_image is not None:
            self.simple_orig_panel.set_image(self.original_image)
            self.adv_orig_panel.set_image(self.original_image)
        else:
            self.simple_orig_panel.clear_image()
            self.adv_orig_panel.clear_image()

        if self.processed_image is not None:
            self.simple_proc_panel.set_image(self.processed_image)
            self.adv_proc_panel.set_image(self.processed_image)
        else:
            self.simple_proc_panel.clear_image()
            self.adv_proc_panel.clear_image()

    # --------------------------------------------------------
    # Status helpers
    # --------------------------------------------------------
    def _set_status(self, msg: str):
        self.simple_status.setText(msg)
        self.adv_status.setText(msg)

    def _check_image_loaded(self) -> bool:
        if self.original_image is None:
            QMessageBox.warning(
                self, "No Image", "Please upload a photo first."
            )
            return False
        return True

    def _check_processed_exists(self) -> bool:
        if self.processed_image is None:
            QMessageBox.warning(
                self,
                "No Processed Image",
                "Please process the image first before performing this action.",
            )
            return False
        return True

    # --------------------------------------------------------
    # Common handlers
    # --------------------------------------------------------
    def _on_upload(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif);;All Files (*)",
        )
        if not file_path:
            return

        if not validate_file_type(file_path):
            QMessageBox.warning(
                self, "Invalid File", "Unsupported file format."
            )
            return

        image = load_image(file_path)
        if image is None:
            QMessageBox.warning(
                self, "Error", "Failed to load the image."
            )
            return

        self.original_image = image
        self.processed_image = None
        self.image_path = file_path

        # Update info + reset state
        info = get_image_info(image)
        self.adv_size_value.setText(
            f"{info['width']} x {info['height']} ({info['channels']} channel"
            f"{'s' if info['channels'] > 1 else ''})"
        )
        self.current_step_index = 0
        self.last_analysis = None
        self._clear_analysis_displays()
        self._refresh_step_highlights()
        self._refresh_displays()
        self.simple_analysis_box.setVisible(False)
        self._set_status("Photo loaded successfully.")

    def _on_save(self):
        if not self._check_processed_exists():
            return

        suggested = "restored_image.png"
        if self.image_path:
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            suggested = f"{base}_restored.png"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            suggested,
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
        )
        if not file_path:
            return

        if save_image(self.processed_image, file_path):
            self._set_status(
                f"Image saved successfully to: {os.path.basename(file_path)}"
            )
        else:
            QMessageBox.warning(self, "Error", "Failed to save the image.")

    # --------------------------------------------------------
    # Simple Mode handlers
    # --------------------------------------------------------
    def _on_enhance_simple(self):
        if not self._check_image_loaded():
            return

        self._set_status("Enhancing photo...")
        # Run the full pipeline in one shot
        self.processed_image = restore_photo(self.original_image)
        self._refresh_displays()

        # Reset advanced step tracker since the image now reflects a one-shot result
        self.current_step_index = len(ADVANCED_STEPS)
        self._refresh_step_highlights()

        # Clear cached analysis so View Analysis recomputes against the new image
        self.last_analysis = None
        self._clear_analysis_displays()
        self.simple_analysis_box.setVisible(False)

        self._set_status("Enhancement completed.")

    def _on_view_analysis(self):
        if not self._check_image_loaded():
            return
        if not self._check_processed_exists():
            return

        self.last_analysis = get_full_analysis(
            self.original_image, self.processed_image
        )
        self._apply_analysis_to_labels(self.last_analysis)
        self._draw_histograms(self.last_analysis)

        self.simple_analysis_box.setVisible(True)
        self._set_status("Analysis ready. See values below.")

    def _on_reset_simple(self):
        """Simple-mode reset: clear the enhanced image, keep the original."""
        self.processed_image = None
        self.last_analysis = None
        self.current_step_index = 0
        self._clear_analysis_displays()
        self.simple_analysis_box.setVisible(False)
        self._refresh_step_highlights()
        self._refresh_displays()
        if self.original_image is None:
            self._set_status("Upload an old photo to begin.")
        else:
            self._set_status("Reset complete. Original photo preserved.")

    # --------------------------------------------------------
    # Advanced Mode handlers
    # --------------------------------------------------------
    def _on_run_current(self):
        if not self._check_image_loaded():
            return
        idx = max(0, min(self.current_step_index, len(ADVANCED_STEPS) - 1))
        self._execute_step(idx)
        # Move pointer forward so subsequent "Run Next" continues from here
        self.current_step_index = min(idx + 1, len(ADVANCED_STEPS))
        self._refresh_step_highlights()

    def _on_run_next(self):
        if not self._check_image_loaded():
            return
        if self.current_step_index >= len(ADVANCED_STEPS):
            self._set_status(
                "All steps completed. Click Reset Advanced Process to restart."
            )
            return
        self._execute_step(self.current_step_index)
        self.current_step_index += 1
        self._refresh_step_highlights()

    def _on_run_all(self):
        if not self._check_image_loaded():
            return
        # Start fresh and run every step in order
        self.processed_image = None
        self.current_step_index = 0
        self._clear_analysis_displays()
        for i in range(len(ADVANCED_STEPS)):
            self._execute_step(i)
            self.current_step_index = i + 1
            self._refresh_step_highlights()
        self._set_status("All steps completed.")

    def _on_reset_advanced(self):
        self.processed_image = None
        self.current_step_index = 0
        self.last_analysis = None
        self._clear_analysis_displays()
        self._refresh_step_highlights()
        self._refresh_displays()
        if self.original_image is None:
            self._set_status("Upload a photo, then run steps to see each effect.")
        else:
            self._set_status("Advanced process reset. Ready to run from step 1.")

    # --------------------------------------------------------
    # Step execution dispatcher
    # --------------------------------------------------------
    def _execute_step(self, index: int):
        """Apply the processing function for the given step index.

        Steps are sequential: each step takes the running processed image as
        input. If no processed image exists yet, the original image is used.
        Steps 'Image Analysis' and 'Histogram' do not transform the image -
        they only update the analysis panel.
        """
        name, step_id = ADVANCED_STEPS[index]
        source = (
            self.processed_image
            if self.processed_image is not None
            else self.original_image
        )

        if step_id == "grayscale":
            self.processed_image = convert_to_grayscale(source)

        elif step_id == "normalize":
            self.processed_image = normalize_intensity(source)

        elif step_id == "contrast":
            self.processed_image = clahe_enhancement(source)

        elif step_id == "denoise":
            self.processed_image = median_filter(source, kernel_size=3)

        elif step_id == "sharpen":
            self.processed_image = sharpen_image(source, strength=0.8)

        elif step_id == "final":
            # Brightness/contrast polish - the final touch in the pipeline
            self.processed_image = adjust_brightness_contrast(
                source, brightness=5, contrast=1.1
            )

        elif step_id == "analyze":
            self.last_analysis = get_full_analysis(
                self.original_image, self.processed_image
            )
            self._apply_analysis_to_labels(self.last_analysis)

        elif step_id == "histogram":
            if self.last_analysis is None:
                self.last_analysis = get_full_analysis(
                    self.original_image, self.processed_image
                )
                self._apply_analysis_to_labels(self.last_analysis)
            self._draw_histograms(self.last_analysis)

        self.current_step_label.setText(f"Current Step: {name}")
        self._refresh_displays()
        self._set_status(f"Step '{name}' applied.")

    def _refresh_step_highlights(self):
        """Update the visual state of each step item in the advanced sidebar."""
        for i, lbl in enumerate(self.step_item_labels):
            if i < self.current_step_index:
                lbl.setObjectName("stepItemDone")
            elif i == self.current_step_index:
                lbl.setObjectName("stepItemActive")
            else:
                lbl.setObjectName("stepItem")
            # Force QSS to re-evaluate after objectName change
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

        if self.current_step_index >= len(ADVANCED_STEPS):
            self.current_step_label.setText("Current Step: All steps completed")
        elif self.original_image is None:
            self.current_step_label.setText("Current Step: -")
        else:
            next_name = ADVANCED_STEPS[self.current_step_index][0]
            self.current_step_label.setText(f"Next Step: {next_name}")

    # --------------------------------------------------------
    # Analysis display helpers
    # --------------------------------------------------------
    def _apply_analysis_to_labels(self, analysis: dict):
        mean_val = analysis["processed_mean"]
        std_val = analysis["processed_std"]
        corr_val = analysis["correlation"]

        self.simple_mean_value.setText(f"{mean_val:.2f}")
        self.simple_std_value.setText(f"{std_val:.2f}")
        self.simple_corr_value.setText(f"{corr_val:.4f}")

        self.adv_mean_value.setText(f"{mean_val:.2f}")
        self.adv_std_value.setText(f"{std_val:.2f}")
        self.adv_corr_value.setText(f"{corr_val:.4f}")

    def _clear_analysis_displays(self):
        for v in (
            self.simple_mean_value,
            self.simple_std_value,
            self.simple_corr_value,
            self.adv_mean_value,
            self.adv_std_value,
            self.adv_corr_value,
        ):
            v.setText("--")
        # Clear histogram canvas
        self.adv_hist_figure.clear()
        self.adv_hist_canvas.draw()

    def _draw_histograms(self, analysis: dict):
        self.adv_hist_figure.clear()
        ax1 = self.adv_hist_figure.add_subplot(121)
        ax2 = self.adv_hist_figure.add_subplot(122)

        for ax in (ax1, ax2):
            ax.set_facecolor("#181825")
            ax.tick_params(colors="#cdd6f4", labelsize=7)
            ax.spines["bottom"].set_color("#45475a")
            ax.spines["left"].set_color("#45475a")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        orig_hist = analysis.get("original_histogram", [])
        if orig_hist:
            if len(orig_hist) == 1:
                ax1.plot(orig_hist[0], color="#89b4fa", linewidth=0.8)
                ax1.fill_between(
                    range(256), orig_hist[0], alpha=0.3, color="#89b4fa"
                )
            else:
                colors = ["#89b4fa", "#a6e3a1", "#f38ba8"]
                for i, h in enumerate(orig_hist):
                    ax1.plot(h, color=colors[i], linewidth=0.7, alpha=0.85)
        ax1.set_title("Original", color="#cdd6f4", fontsize=9)
        ax1.set_xlim([0, 256])

        proc_hist = analysis.get("processed_histogram", [])
        if proc_hist:
            if len(proc_hist) == 1:
                ax2.plot(proc_hist[0], color="#a6e3a1", linewidth=0.8)
                ax2.fill_between(
                    range(256), proc_hist[0], alpha=0.3, color="#a6e3a1"
                )
            else:
                colors = ["#89b4fa", "#a6e3a1", "#f38ba8"]
                for i, h in enumerate(proc_hist):
                    ax2.plot(h, color=colors[i], linewidth=0.7, alpha=0.85)
        ax2.set_title("Processed", color="#cdd6f4", fontsize=9)
        ax2.set_xlim([0, 256])

        self.adv_hist_figure.tight_layout(pad=1.0)
        self.adv_hist_canvas.draw()
