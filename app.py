
from pathlib import Path
import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# 1. PAGE CONFIGURATION AND DASHBOARD STYLE
# ============================================================

st.set_page_config(
    page_title="EcoBuild | Structural Crack Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.85;
    }

    .section-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.75;
    }

    .small-note {
        font-size: 0.85rem;
        opacity: 0.8;
    }
    </style>

    <div class="hero">
        <div class="section-label">Civil Engineering • AI • Sustainability</div>
        <div class="hero-title">🏗️ EcoBuild</div>
        <div class="hero-subtitle">
            AI-Based Concrete Crack Detection and Severity Assessment
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Analyse concrete surface images, inspect predicted crack regions, "
    "estimate image-based geometry, and export results."
)

st.warning(
    "Research prototype only. AI predictions and image measurements "
    "cannot independently establish structural safety or determine "
    "a suitable repair."
)


# ============================================================
# 2. MODEL PATH AND LOADING
# ============================================================

MODEL_PATH = Path(
    "runs/segment/runs/crack_project/"
    "crack_segmentation/weights/best.pt"
)


@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


if not MODEL_PATH.exists():
    st.error("The trained model could not be found.")
    st.code(str(MODEL_PATH))
    st.write(
        "Check the model path and run Streamlit from your "
        "ConcreteCrackAI project folder."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()


# ============================================================
# 3. SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Analysis settings")

    confidence_threshold = st.slider(
        "Detection confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Increasing this threshold filters out more predictions.",
    )

    use_calibration = st.checkbox(
        "Enable physical scale calibration",
        value=False,
    )

    pixels_per_mm = None

    if use_calibration:
        pixels_per_mm = st.number_input(
            "Pixels per millimetre",
            min_value=0.1,
            value=10.0,
            step=0.5,
            help=(
                "Use a scale determined from a known-size reference "
                "in the same plane as the crack."
            ),
        )

    st.divider()

    st.markdown("**Project information**")
    st.caption("Project: EcoBuild")
    st.caption("Task: Concrete crack segmentation")
    st.caption("Framework: Ultralytics YOLO + Streamlit")
    st.caption("Inference: CPU or supported device")

    st.divider()

    st.caption(
        "Measurements depend on image quality, calibration and "
        "segmentation accuracy."
    )


# ============================================================
# 4. IMAGE AND GEOMETRY FUNCTIONS
# ============================================================

def create_instance_masks(result, image_shape):
    """Resize predicted instance masks to the uploaded image dimensions."""
    height, width = image_shape[:2]
    instance_masks = []

    if result.masks is None or result.masks.data is None:
        return instance_masks

    for mask in result.masks.data.cpu().numpy():
        resized = cv2.resize(
            mask.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        instance_masks.append((resized > 0.5).astype(np.uint8))

    return instance_masks


def combine_masks(instance_masks, image_shape):
    """Create a union mask so overlapping regions are not double-counted."""
    height, width = image_shape[:2]
    combined = np.zeros((height, width), dtype=np.uint8)

    for mask in instance_masks:
        combined = np.maximum(combined, mask)

    return combined


def calculate_geometry(binary_mask):
    """
    Estimate mask area, approximate skeleton length and local widths.
    Measurements are in pixels.
    """
    mask = binary_mask.astype(np.uint8)
    area_px2 = int(np.count_nonzero(mask))
    skeleton = np.zeros_like(mask)

    if area_px2 == 0:
        return {
            "area_px2": 0,
            "length_px": 0.0,
            "mean_width_px": 0.0,
            "median_width_px": 0.0,
            "p95_width_px": 0.0,
            "max_width_px": 0.0,
            "skeleton": skeleton,
        }

    # Morphological skeleton approximation.
    work = mask.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS, (3, 3)
    )

    while cv2.countNonZero(work) > 0:
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        residue = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        work = eroded

    sk = skeleton > 0

    # Approximate length by counting neighbouring skeleton connections.
    horizontal = np.count_nonzero(sk[:, :-1] & sk[:, 1:])
    vertical = np.count_nonzero(sk[:-1, :] & sk[1:, :])
    diagonal_1 = np.count_nonzero(sk[:-1, :-1] & sk[1:, 1:])
    diagonal_2 = np.count_nonzero(sk[:-1, 1:] & sk[1:, :-1])

    length_px = float(
        horizontal
        + vertical
        + np.sqrt(2) * (diagonal_1 + diagonal_2)
    )

    if length_px == 0:
        length_px = float(np.count_nonzero(sk))

    # Distance transform approximates the local radius within the mask.
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    widths = distance[sk] * 2.0

    if widths.size == 0:
        widths = distance[mask > 0] * 2.0

    return {
        "area_px2": area_px2,
        "length_px": length_px,
        "mean_width_px": float(np.mean(widths)),
        "median_width_px": float(np.median(widths)),
        "p95_width_px": float(np.percentile(widths, 95)),
        "max_width_px": float(np.max(widths)),
        "skeleton": skeleton,
    }


def convert_measurements(geometry, calibrated, scale):
    """Convert pixel measurements when a physical calibration is supplied."""
    if calibrated and scale and scale > 0:
        return {
            "area": geometry["area_px2"] / (scale ** 2),
            "length": geometry["length_px"] / scale,
            "mean_width": geometry["mean_width_px"] / scale,
            "median_width": geometry["median_width_px"] / scale,
            "p95_width": geometry["p95_width_px"] / scale,
            "max_width": geometry["max_width_px"] / scale,
            "area_unit": "mm^2",
            "length_unit": "mm",
            "width_unit": "mm",
        }

    return {
        "area": float(geometry["area_px2"]),
        "length": geometry["length_px"],
        "mean_width": geometry["mean_width_px"],
        "median_width": geometry["median_width_px"],
        "p95_width": geometry["p95_width_px"],
        "max_width": geometry["max_width_px"],
        "area_unit": "px^2",
        "length_unit": "px",
        "width_unit": "px",
    }


# ============================================================
# 5. IMAGE UPLOAD
# ============================================================

st.markdown("### 01  |  Upload and analyse")

uploaded_file = st.file_uploader(
    "Choose a concrete surface image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Use a clear image with visible concrete surface details.",
)

if uploaded_file is None:
    st.info(
        "Upload an image to start the analysis dashboard."
    )

    st.markdown("### How EcoBuild works")

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        """
        **1. Image input**

        Upload a concrete surface photograph.
        """
    )

    c2.markdown(
        """
        **2. AI segmentation**

        The trained model predicts potential crack regions.
        """
    )

    c3.markdown(
        """
        **3. Results**

        Review image-based measurements and export the report.
        """
    )

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        with st.spinner("Running crack segmentation..."):
            results = model.predict(
                source=image_array,
                conf=confidence_threshold,
                imgsz=640,
                verbose=False,
            )

        result = results[0]

        annotated_rgb = cv2.cvtColor(
            result.plot(), cv2.COLOR_BGR2RGB
        )

        instance_masks = create_instance_masks(
            result, image_array.shape
        )

        combined_mask = combine_masks(
            instance_masks, image_array.shape
        )

        geometry = calculate_geometry(combined_mask)

        measurements = convert_measurements(
            geometry,
            use_calibration,
            pixels_per_mm,
        )

        detection_count = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        # ====================================================
        # 6. DASHBOARD METRICS
        # ====================================================

        st.divider()
        st.markdown("### 02  |  Analysis dashboard")

        st.caption(f"Image: {uploaded_file.name}")

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Crack instances",
            detection_count,
        )

        k2.metric(
            f"Mask area ({measurements['area_unit']})",
            f"{measurements['area']:,.2f}",
        )

        k3.metric(
            f"Estimated length ({measurements['length_unit']})",
            f"{measurements['length']:,.2f}",
        )

        k4.metric(
            f"Mean width ({measurements['width_unit']})",
            f"{measurements['mean_width']:.2f}",
        )

        if detection_count == 0 or geometry["area_px2"] == 0:
            st.warning(
                "No crack mask was produced at this confidence threshold. "
                "This does not prove that the concrete is crack-free."
            )

        # ====================================================
        # 7. IMAGE COMPARISON
        # ====================================================

        st.divider()
        st.markdown("### 03  |  Visual inspection")

        image_tab, mask_tab, skeleton_tab = st.tabs(
            ["Original vs prediction", "Crack mask", "Centreline"]
        )

        with image_tab:
            left, right = st.columns(2)

            with left:
                st.markdown("**Original image**")
                st.image(image, width="stretch")

            with right:
                st.markdown("**AI prediction overlay**")
                st.image(annotated_rgb, width="stretch")

        with mask_tab:
            st.caption(
                "White pixels indicate the combined predicted crack mask."
            )
            st.image(
                combined_mask * 255,
                clamp=True,
                width="stretch",
            )

        with skeleton_tab:
            st.caption(
                "The thin line is used for approximate geometry calculations."
            )
            st.image(
                geometry["skeleton"] * 255,
                clamp=True,
                width="stretch",
            )

        # ====================================================
        # 8. INDIVIDUAL DETECTIONS
        # ====================================================

        st.divider()
        st.markdown("### 04  |  Individual crack detections")

        crack_rows = []

        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                    .tolist()
                )

                row = {
                    "Crack ID": i + 1,
                    "Confidence (%)": round(confidence * 100, 2),
                    "Box width (px)": max(0, x2 - x1),
                    "Box height (px)": max(0, y2 - y1),
                }

                if i < len(instance_masks):
                    row["Mask area (px^2)"] = int(
                        np.count_nonzero(instance_masks[i])
                    )

                crack_rows.append(row)

        if crack_rows:
            crack_df = pd.DataFrame(crack_rows)

            st.dataframe(
                crack_df,
                hide_index=True,
                width="stretch",
            )

            st.caption(
                "Confidence is a model score, not a probability of "
                "structural safety. Box dimensions are image coordinates."
            )
        else:
            st.info("No individual detections to display.")

        # ====================================================
        # 9. GEOMETRY DETAILS
        # ====================================================

        st.divider()
        st.markdown("### 05  |  Crack geometry")

        g1, g2, g3 = st.columns(3)

        g1.metric(
            f"Median width ({measurements['width_unit']})",
            f"{measurements['median_width']:.2f}",
        )

        g2.metric(
            f"95th-percentile width ({measurements['width_unit']})",
            f"{measurements['p95_width']:.2f}",
        )

        g3.metric(
            f"Maximum width ({measurements['width_unit']})",
            f"{measurements['max_width']:.2f}",
        )

        if use_calibration:
            st.success(
                "Calibration is enabled. Physical estimates are only "
                "meaningful if the reference scale is accurate and "
                "in the same plane as the crack."
            )
        else:
            st.info(
                "Calibration is disabled. Measurements are in pixels, "
                "not millimetres."
            )

        # ====================================================
        # 10. SEVERITY GUIDANCE
        # ====================================================

        st.divider()
        st.markdown("### 06  |  Severity guidance")

        if geometry["area_px2"] == 0:
            st.info(
                "No predicted crack mask is available for width guidance."
            )

        elif not use_calibration:
            st.info(
                "Enable a valid physical scale calibration before using "
                "the optional width-based demonstration bands."
            )

        else:
            st.caption(
                "These configurable bands are for demonstration only. "
                "They are not universal engineering thresholds."
            )

            s1, s2 = st.columns(2)

            with s1:
                threshold_1 = st.number_input(
                    "Lower band threshold (mm)",
                    min_value=0.01,
                    value=0.30,
                    step=0.05,
                    key="band_low",
                )

            with s2:
                threshold_2 = st.number_input(
                    "Upper band threshold (mm)",
                    min_value=0.02,
                    value=0.40,
                    step=0.05,
                    key="band_high",
                )

            if threshold_2 <= threshold_1:
                st.error(
                    "The upper band threshold must exceed the lower one."
                )
            else:
                width_mm = measurements["max_width"]

                if width_mm < threshold_1:
                    category = "Lower measured-width band"
                elif width_mm < threshold_2:
                    category = "Intermediate measured-width band"
                else:
                    category = "Higher measured-width band"

                st.subheader(category)

                st.write(
                    "This category uses the maximum estimated image width "
                    "and your selected thresholds. It is not a structural "
                    "safety rating."
                )

        # ====================================================
        # 11. MAINTENANCE GUIDANCE
        # ====================================================

        st.divider()
        st.markdown("### 07  |  Maintenance guidance")

        with st.expander("View recommended inspection actions", expanded=True):
            st.markdown(
                """
                **Document the crack:** Record the location, direction,
                date and appearance. Save photographs for comparison.

                **Monitor changes:** Revisit the same area and use a
                consistent method to observe changes over time.

                **Check surrounding conditions:** Look for leakage,
                corrosion staining, exposed reinforcement, displacement
                or visible deformation.

                **Seek professional assessment:** Growing cracks, diagonal
                cracks near supports, exposed reinforcement or visible
                deformation warrant appropriate engineering inspection.

                **Do not select repairs from AI output alone:** The cause,
                structural role and site conditions must be assessed.
                """
            )

        # ====================================================
        # 12. EXPORTS
        # ====================================================

        st.divider()
        st.markdown("### 08  |  Export results")

        report = {
            "image_name": uploaded_file.name,
            "detected_instances": detection_count,
            "confidence_threshold": confidence_threshold,
            "mask_area": measurements["area"],
            "area_unit": measurements["area_unit"],
            "centreline_length": measurements["length"],
            "length_unit": measurements["length_unit"],
            "mean_width": measurements["mean_width"],
            "median_width": measurements["median_width"],
            "p95_width": measurements["p95_width"],
            "maximum_width": measurements["max_width"],
            "width_unit": measurements["width_unit"],
            "calibration_used": use_calibration,
            "pixels_per_mm": (
                pixels_per_mm if use_calibration else ""
            ),
        }

        report_df = pd.DataFrame([report])

        csv_bytes = report_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        annotated_buffer = io.BytesIO()
        Image.fromarray(annotated_rgb).save(
            annotated_buffer,
            format="PNG",
        )

        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "Download measurement report (CSV)",
                data=csv_bytes,
                file_name="ecobuild_crack_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with d2:
            st.download_button(
                "Download annotated image (PNG)",
                data=annotated_buffer.getvalue(),
                file_name="ecobuild_prediction.png",
                mime="image/png",
                use_container_width=True,
            )

        if crack_rows:
            individual_bytes = pd.DataFrame(
                crack_rows
            ).to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "Download individual detections (CSV)",
                data=individual_bytes,
                file_name="ecobuild_individual_cracks.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as exc:
        st.error(f"An error occurred during analysis: {exc}")
        st.exception(exc)


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "EcoBuild | AI-based concrete crack detection prototype. "
    "Image measurements are approximate. The application does not "
    "certify structural safety or replace professional inspection."
)
