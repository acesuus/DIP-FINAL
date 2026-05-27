# MemoryFix: Offline Old Photo Restoration and Enhancement Tool

## Project Context

**MemoryFix** is an offline desktop application designed to restore and enhance old photos using traditional Digital Image Processing techniques. The project focuses on scanned hard-copy photos and old digital images that may look faded, blurry, noisy, dark, or low in contrast.

The application allows the user to import an old photo, apply restoration and enhancement methods, compare the original and processed output, view basic image analysis values, and save the final restored image.

This project is built for a Digital Image Processing final project. It follows the required core components of the subject, including:

- Contrast enhancement
- Image filtering
- Image restoration
- Image analysis

The application must work fully offline and should not use cloud APIs, AI image generation, or online services.

---

## Project Title

**MemoryFix: Offline Old Photo Restoration and Enhancement Tool**

---

## Main Objective

To develop a modern offline desktop GUI application that restores and enhances old photos using Python, OpenCV, NumPy, and PyQt.

---

## Target Users

- Students working on a Digital Image Processing project
- Users who want to restore old family photos
- Users with scanned old photos or old digital images
- Users who need a simple offline image enhancement tool

---

## Tech Stack

### Programming Language

- Python 3.10 or higher

### GUI Framework

- PyQt6

### Image Processing

- OpenCV
- NumPy
- SciPy

### Visualization

- Matplotlib

### Image File Handling

- Pillow

---

## Main Features

### 1. Import Old Photo

The user can upload an old image from their local computer.

Supported formats:

- JPG
- JPEG
- PNG
- BMP
- TIFF

The uploaded image should be shown in the **Original Image** preview panel.

---

### 2. Image Preview and Comparison

The GUI must show two main panels:

```text
Original Image | Restored Image
```

This allows the user to compare the image before and after processing.

---

### 3. Preprocessing

Preprocessing prepares the image before enhancement and restoration.

Required preprocessing functions:

- Convert image to grayscale
- Normalize image intensity
- Prepare image format for OpenCV processing and PyQt display

Related DIP topic:

- Image transformation

---

### 4. Contrast Enhancement

This feature improves faded or low-contrast old photos.

Required methods:

- Contrast stretching
- Histogram equalization

Recommended method:

- CLAHE, or Contrast Limited Adaptive Histogram Equalization

Purpose:

- Improve faded areas
- Make dark and bright regions more balanced
- Improve visibility of faces and objects

---

### 5. Noise Reduction

This feature reduces unwanted grain, small marks, and noise.

Required method:

- Median filtering

Optional methods:

- Gaussian filtering
- Bilateral filtering

Purpose:

- Reduce small dust-like marks
- Smooth grainy areas
- Keep the image natural-looking

---

### 6. Sharpening

This feature improves the clarity of the image.

Required method:

- Sharpening using kernel convolution

Example sharpening kernel:

```python
[[ 0, -1,  0],
 [-1,  5, -1],
 [ 0, -1,  0]]
```

Purpose:

- Make facial details clearer
- Improve object edges
- Improve overall image sharpness

---

### 7. Basic Photo Restoration

This is the main feature of the application.

The restoration process should combine:

1. Denoising
2. Contrast enhancement
3. Sharpening
4. Brightness and contrast adjustment

Suggested restoration pipeline:

```text
Load image
↓
Convert to proper color format
↓
Apply denoising
↓
Apply contrast enhancement
↓
Apply sharpening
↓
Adjust brightness and contrast
↓
Display restored output
```

---

### 8. Image Analysis

The system should calculate and display simple image statistics.

Required values:

- Mean
- Standard deviation
- Correlation coefficient between original and restored image

Additional output:

- Histogram before enhancement
- Histogram after enhancement

Purpose:

- Show measurable difference between original and restored image
- Support the project report and presentation

---

### 9. Save Restored Image

The user can save the final restored image.

Supported output formats:

- JPG
- PNG

Important rule:

- The original image must not be overwritten.

---

## Modern GUI Layout

The interface should be simple, clean, and modern.

### Main Window Sections

```text
---------------------------------------------------------
| Sidebar | Original Image Preview | Restored Image Preview |
|         |                        |                        |
|         |                        |                        |
|-------------------------------------------------------|
| Bottom Status / Image Analysis Panel                  |
---------------------------------------------------------
```

---

### Sidebar Buttons

Include these buttons:

- Import Image
- Preprocess
- Enhance Contrast
- Reduce Noise
- Sharpen Image
- Restore Photo
- Analyze Image
- Save Output
- Reset

---

### Control Panel

Include these controls:

- Brightness slider
- Contrast slider
- Sharpness slider
- Filter strength slider
- Filter type dropdown
- Apply button

---

### Bottom Analysis Panel

Display:

- Image width and height
- Mean value
- Standard deviation
- Correlation coefficient
- Current process status

Example status messages:

```text
Image loaded successfully.
Contrast enhancement applied.
Noise reduction completed.
Restoration completed.
Image saved successfully.
```

---

## Recommended Project Folder Structure

```text
MemoryFix/
│
├── main.py
├── gui.py
├── image_processor.py
├── analysis.py
├── utils.py
├── requirements.txt
├── README.md
│
├── assets/
│   └── icons/
│
├── outputs/
│
└── sample_images/
```

---

## File Responsibilities

### main.py

Starts the application.

Responsibilities:

- Create the QApplication
- Load the main window
- Run the event loop

---

### gui.py

Contains the PyQt6 interface.

Responsibilities:

- Main window layout
- Buttons
- Image preview panels
- Sliders and controls
- Status and analysis labels
- Button event connections

---

### image_processor.py

Contains the image processing functions.

Responsibilities:

- Load image
- Convert image formats
- Grayscale conversion
- Normalization
- Contrast stretching
- Histogram equalization
- Median filtering
- Sharpening
- Main restoration pipeline
- Save image

---

### analysis.py

Contains image analysis functions.

Responsibilities:

- Calculate mean
- Calculate standard deviation
- Calculate correlation coefficient
- Generate histogram data

---

### utils.py

Contains helper functions.

Responsibilities:

- Convert OpenCV image to PyQt image
- Resize image for display
- Validate file types
- Handle output paths

---

## Step-by-Step Implementation Plan

### Step 1: Initialize Project

Create the folder structure and required files.

Deliverables:

- Project folders created
- Empty Python files created
- `requirements.txt` completed

---

### Step 2: Install Dependencies

Add this to `requirements.txt`:

```text
opencv-python
numpy
scipy
matplotlib
PyQt6
Pillow
```

---

### Step 3: Create Basic PyQt Window

Implement the main window with:

- Sidebar
- Original image panel
- Restored image panel
- Bottom status panel

Deliverable:

- A running PyQt6 window with modern layout

---

### Step 4: Add Image Import

Implement image upload using QFileDialog.

Deliverables:

- User can select an image
- Image appears in the original preview panel
- Image path is stored
- Original OpenCV image is stored

---

### Step 5: Add OpenCV-to-PyQt Image Display

Implement a helper function that converts an OpenCV image to QPixmap.

Deliverable:

- Images display correctly in the GUI without wrong colors

---

### Step 6: Add Preprocessing

Implement:

- Grayscale conversion
- Normalization

Deliverable:

- User can click Preprocess and see the processed image

---

### Step 7: Add Contrast Enhancement

Implement:

- Contrast stretching
- Histogram equalization
- CLAHE if possible

Deliverable:

- User can apply contrast enhancement to faded photos

---

### Step 8: Add Noise Reduction

Implement:

- Median filtering

Optional:

- Gaussian filtering
- Bilateral filtering

Deliverable:

- User can reduce image noise and grain

---

### Step 9: Add Sharpening

Implement sharpening using kernel convolution.

Deliverable:

- User can make image details clearer

---

### Step 10: Add Main Restore Photo Button

Create one main restoration function that applies:

1. Denoising
2. Contrast enhancement
3. Sharpening
4. Brightness and contrast adjustment

Deliverable:

- User can restore a photo with one button

---

### Step 11: Add Image Analysis

Implement and display:

- Mean
- Standard deviation
- Correlation coefficient

Deliverable:

- Analysis values are shown in the bottom panel

---

### Step 12: Add Histogram Display

Use Matplotlib to display:

- Histogram of original image
- Histogram of restored image

Deliverable:

- User can view histogram comparison

---

### Step 13: Add Save Output

Implement save function.

Deliverable:

- User can save restored image as JPG or PNG

---

### Step 14: Add Reset Function

Reset should:

- Clear processed image
- Clear analysis values
- Reset sliders
- Restore original image state

---

### Step 15: Improve UI Design

Improve the layout using:

- Modern spacing
- Rounded buttons using QSS
- Dark or light theme
- Clear labels
- Better status messages

Deliverable:

- GUI looks modern and presentation-ready

---

## Minimum Viable Product

The project is considered working when it can:

1. Import an old image
2. Display the original image
3. Convert to grayscale
4. Enhance contrast
5. Reduce noise
6. Sharpen image
7. Restore photo using one button
8. Show mean and standard deviation
9. Show correlation coefficient
10. Save the restored image

---

## Expected Output Images for Report

Include these screenshots or saved outputs in the project report:

1. Original old photo
2. Grayscale image
3. Contrast-enhanced image
4. Noise-reduced image
5. Sharpened image
6. Final restored image
7. Before-and-after comparison
8. Histogram before enhancement
9. Histogram after enhancement
10. Image analysis result panel

---

# Agentic Coding Flow Prompt Template

Use this section to guide an AI coding agent for rapid implementation.

---

## General Instruction for the Coding Agent

You are building a Python desktop application called **MemoryFix: Offline Old Photo Restoration and Enhancement Tool**.

The application must be implemented using:

- Python
- PyQt6
- OpenCV
- NumPy
- SciPy
- Matplotlib
- Pillow

The application must work fully offline. Do not use cloud APIs, online services, AI image generation, or external web requests.

Prioritize a working project first before adding optional polish. Keep the code modular, readable, and beginner-friendly.

---

## Prompt 1: Create the Project Structure

```text
Create a Python project named MemoryFix.

Generate the following files and folders:

MemoryFix/
├── main.py
├── gui.py
├── image_processor.py
├── analysis.py
├── utils.py
├── requirements.txt
├── README.md
├── assets/
│   └── icons/
├── outputs/
└── sample_images/

Add the required dependencies in requirements.txt:
opencv-python
numpy
scipy
matplotlib
PyQt6
Pillow

Do not implement the full app yet. Just create the structure and basic placeholder files.
```

---

## Prompt 2: Build the Main PyQt6 GUI

```text
Implement the main PyQt6 GUI for MemoryFix.

Requirements:
- Use PyQt6.
- Create a modern main window.
- Add a left sidebar with buttons:
  Import Image
  Preprocess
  Enhance Contrast
  Reduce Noise
  Sharpen Image
  Restore Photo
  Analyze Image
  Save Output
  Reset

- Add two image preview panels:
  Original Image
  Restored Image

- Add a bottom panel for:
  Image size
  Mean
  Standard deviation
  Correlation coefficient
  Status message

- Use clean spacing and readable labels.
- Apply simple QSS styling for a modern look.
- Make the GUI run from main.py.

Do not implement image processing yet. Use placeholder button functions.
```

---

## Prompt 3: Implement Image Loading and Display

```text
Implement image loading in MemoryFix.

Requirements:
- Use QFileDialog to let the user import JPG, JPEG, PNG, BMP, or TIFF images.
- Load the image using OpenCV.
- Store the original image as self.original_image.
- Store the current processed image as self.processed_image.
- Convert the OpenCV image from BGR to RGB for display.
- Display the loaded image in the Original Image panel.
- Show image width and height in the bottom panel.
- Update the status message to "Image loaded successfully."

Also implement a helper function in utils.py:
cv_image_to_qpixmap(image)

This function must correctly convert grayscale and RGB images to QPixmap.
```

---

## Prompt 4: Implement Preprocessing

```text
Implement preprocessing for MemoryFix.

Requirements:
- Add functions in image_processor.py:
  convert_to_grayscale(image)
  normalize_intensity(image)

- The Preprocess button should:
  1. Convert the original image to grayscale.
  2. Normalize intensity.
  3. Display the result in the Restored Image panel.
  4. Store the result as self.processed_image.
  5. Update the status message.

Make sure the function works for both color and grayscale images.
```

---

## Prompt 5: Implement Contrast Enhancement

```text
Implement contrast enhancement for MemoryFix.

Requirements:
- Add functions in image_processor.py:
  contrast_stretch(image)
  histogram_equalization(image)
  clahe_enhancement(image)

- The Enhance Contrast button should apply CLAHE by default.
- If the image is color, apply CLAHE to the luminance channel only using LAB color space.
- If the image is grayscale, apply CLAHE directly.
- Display the result in the Restored Image panel.
- Store the result as self.processed_image.
- Update the status message.

Keep the result natural-looking and avoid overprocessing.
```

---

## Prompt 6: Implement Noise Reduction

```text
Implement noise reduction for MemoryFix.

Requirements:
- Add functions in image_processor.py:
  median_filter(image, kernel_size=3)
  gaussian_filter(image, kernel_size=5)
  bilateral_filter(image)

- The Reduce Noise button should apply median filtering by default.
- Use odd kernel sizes only.
- Display the result in the Restored Image panel.
- Store the result as self.processed_image.
- Update the status message.

The goal is to reduce grain and small dust-like noise in old photos.
```

---

## Prompt 7: Implement Sharpening

```text
Implement image sharpening for MemoryFix.

Requirements:
- Add function in image_processor.py:
  sharpen_image(image)

Use a kernel convolution sharpening method.

Suggested kernel:
[[ 0, -1,  0],
 [-1,  5, -1],
 [ 0, -1,  0]]

The Sharpen Image button should:
- Apply sharpening to the current processed image if available.
- Otherwise, apply it to the original image.
- Display the result in the Restored Image panel.
- Store the result as self.processed_image.
- Update the status message.

Avoid changing the original image variable.
```

---

## Prompt 8: Implement Main Restore Photo Pipeline

```text
Implement the main Restore Photo function for MemoryFix.

Requirements:
- Add function in image_processor.py:
  restore_photo(image)

The function should apply this pipeline:
1. Denoising using median or bilateral filtering
2. Contrast enhancement using CLAHE
3. Sharpening using kernel convolution
4. Slight brightness and contrast adjustment

The Restore Photo button should:
- Apply restore_photo() to the original image.
- Display the final restored output in the Restored Image panel.
- Store the result as self.processed_image.
- Update the status message to "Photo restoration completed."

Make the result natural and not overly edited.
```

---

## Prompt 9: Implement Image Analysis

```text
Implement image analysis for MemoryFix.

Requirements:
- Add functions in analysis.py:
  calculate_mean(image)
  calculate_std(image)
  calculate_correlation(original, processed)

The Analyze Image button should:
- Use the original image and processed image.
- Calculate mean of the processed image.
- Calculate standard deviation of the processed image.
- Calculate correlation coefficient between the original and processed image.
- Display the results in the bottom panel.
- Update the status message.

Make sure original and processed images are resized or converted properly before calculating correlation.
```

---

## Prompt 10: Implement Histogram Display

```text
Add histogram display for MemoryFix.

Requirements:
- Use Matplotlib.
- Create a function that shows histogram comparison between original and processed images.
- The histogram should show:
  Original image histogram
  Processed image histogram

Add a button or connect it to Analyze Image.
When Analyze Image is clicked, show the histogram in a popup window or embedded Matplotlib canvas.

Make sure the app does not crash if no image is loaded.
```

---

## Prompt 11: Implement Save Output

```text
Implement saving restored images for MemoryFix.

Requirements:
- Add a Save Output button function.
- Let the user choose save location using QFileDialog.
- Save only the processed image.
- Supported formats:
  JPG
  PNG

If no processed image exists, show a warning message.

Do not overwrite the original image automatically.
Update status message after successful saving.
```

---

## Prompt 12: Implement Reset Function

```text
Implement the Reset button.

Requirements:
- Clear the processed image panel.
- Clear analysis results.
- Reset status message.
- Keep the original image loaded.
- Reset sliders and controls if available.

The user should be able to start processing again from the original image.
```

---

## Prompt 13: Add Brightness, Contrast, and Sharpness Controls

```text
Add sliders to the GUI for:
- Brightness
- Contrast
- Sharpness

Requirements:
- Brightness range: -100 to 100
- Contrast range: 50 to 200
- Sharpness range: 0 to 200

Add an Apply Adjustments button.

The adjustment should apply to the current processed image if available, otherwise to the original image.

Display the adjusted result in the Restored Image panel and update self.processed_image.
```

---

## Prompt 14: Add Error Handling and User Warnings

```text
Improve error handling in MemoryFix.

Requirements:
- Show warning if user clicks processing buttons without loading an image.
- Show warning if user clicks Analyze without a processed image.
- Show warning if user clicks Save without a processed image.
- Prevent crashes from invalid file types.
- Use QMessageBox for warnings.
- Keep status messages clear and student-friendly.
```

---

## Prompt 15: Final Polish and Code Cleanup

```text
Finalize the MemoryFix application.

Requirements:
- Clean unused imports.
- Add comments to important functions.
- Make names readable and beginner-friendly.
- Improve QSS styling.
- Make image panels responsive.
- Make sure all buttons work.
- Test the full flow:
  Import image
  Preprocess
  Enhance contrast
  Reduce noise
  Sharpen
  Restore photo
  Analyze image
  Save output
  Reset

Do not add online features or AI-based processing.
Keep the project simple, offline, and aligned with Digital Image Processing requirements.
```

---

## Recommended Rapid Build Order

For fastest implementation, follow this order:

1. GUI layout
2. Image import and display
3. OpenCV-to-PyQt conversion
4. Preprocessing
5. Contrast enhancement
6. Noise reduction
7. Sharpening
8. Main restore button
9. Analysis values
10. Save output
11. Reset
12. UI polish

---

## Final Acceptance Checklist

The project is complete when:

- The app runs offline
- The user can import an old image
- The original image displays properly
- The processed image displays properly
- Contrast enhancement works
- Noise reduction works
- Sharpening works
- Main restoration pipeline works
- Mean is calculated
- Standard deviation is calculated
- Correlation coefficient is calculated
- Histogram comparison is shown
- Restored image can be saved
- GUI looks clean and modern
- Code is organized into multiple files
