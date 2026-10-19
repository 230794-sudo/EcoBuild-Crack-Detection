
# EcoBuild — AI-Based Concrete Crack Detection

## Project Description
EcoBuild is a machine-learning minor project that uses a YOLO segmentation model to detect and visualise cracks in concrete surface images.

## Features
- Upload concrete images
- Detect and segment crack regions
- Display crack masks and centreline visualisations
- Estimate crack area, length and width statistics in pixels
- Export measurement data as CSV
- Download annotated images

## Requirements
- Python 3.12
- VS Code or another Python environment
- Required Python packages
- Trained model file: best.pt

## How to Run
1. Open the project folder in a terminal.
2. Create and activate a Python virtual environment.
3. Install the required packages.
4. Confirm that the trained model is available at the path configured in app.py.
5. Run:

   streamlit run app.py

6. Open the local URL displayed in the terminal.

## Important Note
The measurements are approximate image-based estimates.
The application is a prototype and is not a substitute for
professional structural inspection.
