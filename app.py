
import os
os.environ["STREAMLIT_SERVER_PORT"] = "8080"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import gc
import time

st.set_page_config(layout="wide", page_title="Trial Mango Lesion Analyzer", page_icon="🥭")
st.title("🥭 Trial Mango Lesion Analyzer")

st.markdown("This is a full-featured trial version for mango lesion analysis, including manual correction, sample management, and CSV export.")

# Constants for trial
TRIAL_CANVAS_SIZE = 400
TRIAL_IMAGE_SIZE = 200
TRIAL_MAX_FILE_SIZE_MB = 5
TRIAL_MAX_SAMPLES = 5

if 'samples' not in st.session_state:
    st.session_state.samples = []
if 'mm_per_px' not in st.session_state:
    st.session_state.mm_per_px = None
if 'canvas_key_counter' not in st.session_state:
    st.session_state.canvas_key_counter = 0
if 'corrected_result' not in st.session_state:
    st.session_state.corrected_result = None
if 'apply_corrections' not in st.session_state:
    st.session_state.apply_corrections = False

def process_uploaded_image(uploaded_file, max_dim=TRIAL_IMAGE_SIZE):
    try:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        scale = min(max_dim / image.height, max_dim / image.width, 1.0)
        if scale < 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        image_np = np.array(image, dtype=np.uint8)
        return image_np, image, scale
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None, None, None

def quick_color_analysis(image_np, mask, mm_per_px):
    if image_np is None or mask is None or mm_per_px is None:
        return 0, 0, 0, None, None
    try:
        if mask.shape != image_np.shape[:2]:
            mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        mango_mask = cv2.inRange(hsv, (20, 40, 60), (85, 255, 255))
        lesion_mask = cv2.inRange(hsv, (8, 50, 30), (25, 255, 140))
        cv2.bitwise_and(mango_mask, mask, mango_mask)
        cv2.bitwise_and(lesion_mask, mask, lesion_mask)
        total_mango_mask = cv2.bitwise_or(mango_mask, lesion_mask)
        mango_area_px = np.count_nonzero(total_mango_mask)
        lesion_area_px = np.count_nonzero(lesion_mask)
        if mango_area_px == 0:
            return 0, 0, 0, None, None
        mm_per_px_sq = mm_per_px * mm_per_px
        mango_area_mm2 = mango_area_px * mm_per_px_sq
        lesion_area_mm2 = lesion_area_px * mm_per_px_sq
        lesion_percent = (lesion_area_mm2 / mango_area_mm2 * 100) if mango_area_mm2 > 0 else 0
        gc.collect()
        return mango_area_mm2, lesion_area_mm2, lesion_percent, total_mango_mask, lesion_mask
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        gc.collect()
        return 0, 0, 0, None, None

uploaded_file = st.file_uploader("Upload mango image (JPG/PNG)", type=["png", "jpg", "jpeg"], help=f"Max size: {TRIAL_MAX_FILE_SIZE_MB}MB")

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_size = len(file_bytes) / (1024 * 1024)
    if file_size > TRIAL_MAX_FILE_SIZE_MB:
        st.error(f"File too large: {file_size:.1f}MB. Use files under {TRIAL_MAX_FILE_SIZE_MB}MB.")
        st.stop()
    image_np, image, scale = process_uploaded_image(uploaded_file)
    if image_np is None:
        st.stop()
    st.image(image_np, caption="Uploaded Image", width=TRIAL_CANVAS_SIZE)
    st.success(f"Image loaded: {image_np.shape[1]}x{image_np.shape[0]} pixels")
    st.markdown("## 1️⃣ Set Scale")
    scale_canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=3,
        stroke_color="rgba(255,0,0,1)",
        background_image=image,
        update_streamlit=True,
        height=image_np.shape[0],
        width=image_np.shape[1],
        drawing_mode="line",
        key="trial_scale_canvas",
    )
    scale_length_mm = st.number_input("Real length of drawn line (mm):", min_value=0.1, value=10.0, step=1.0)
    scale_px = None
    if scale_canvas and hasattr(scale_canvas, 'json_data') and scale_canvas.json_data:
        objects = scale_canvas.json_data.get("objects", [])
        if objects and isinstance(objects, list):
            obj = objects[-1]
            if obj and isinstance(obj, dict) and obj.get("type") == "line":
                try:
                    x1, y1 = float(obj.get("x1", 0)), float(obj.get("y1", 0))
                    x2, y2 = float(obj.get("x2", 0)), float(obj.get("y2", 0))
                    scale_px = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if scale_px > 0:
                        st.success(f"Line drawn: {scale_px:.1f} pixels")
                    else:
                        scale_px = None
                except Exception:
                    scale_px = None
    if scale_px and scale_length_mm > 0:
        st.session_state.mm_per_px = scale_length_mm / scale_px
        st.success(f"Scale set: {st.session_state.mm_per_px:.4f} mm/pixel")
        st.markdown("## 2️⃣ Analyze Mango")
        drawing_mode = st.radio("Drawing mode:", ["circle", "rect", "polygon", "transform"])
        canvas_key = f"trial_mango_canvas_{st.session_state.canvas_key_counter}"
        mango_canvas = st_canvas(
            fill_color="rgba(255,165,0,0.2)",
            stroke_width=3,
            stroke_color="rgba(255,165,0,1)",
            background_image=image,
            update_streamlit=True,
            height=image_np.shape[0],
            width=image_np.shape[1],
            drawing_mode=drawing_mode,
            key=canvas_key,
        )
        process_analysis = False
        if (mango_canvas and hasattr(mango_canvas, 'image_data') and mango_canvas.image_data is not None):
            if (len(mango_canvas.image_data.shape) >= 3 and mango_canvas.image_data.shape[2] >= 4):
                alpha_channel = mango_canvas.image_data[:,:,3]
                has_drawing = np.any(alpha_channel > 0)
                if has_drawing:
                    process_analysis = True
        if process_analysis:
            image_data = mango_canvas.image_data
            alpha_channel = image_data[:,:,3]
            mask = (alpha_channel > 0).astype(np.uint8) * 255
            mango_area_mm2, lesion_area_mm2, lesion_percent, total_mask, lesion_mask = quick_color_analysis(image_np, mask, st.session_state.mm_per_px)
            if mango_area_mm2 > 0:
                st.markdown("### 📊 Analysis Results")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.image(total_mask, caption="🥭 Total Mango Area", width=120)
                with col2:
                    st.image(lesion_mask, caption="🔴 Lesion Areas", width=120)
                with col3:
                    st.metric("Total Area", f"{mango_area_mm2:.1f} mm²")
                    st.metric("Lesion Area", f"{lesion_area_mm2:.1f} mm²")
                    st.metric("Lesion %", f"{lesion_percent:.1f}%")
                result = {
                    "Sample": len(st.session_state.samples) + 1,
                    "Area (mm²)": round(mango_area_mm2, 1),
                    "Lesions (mm²)": round(lesion_area_mm2, 1),
                    "Lesion %": round(lesion_percent, 1)
                }
                st.dataframe(pd.DataFrame([result]), use_container_width=True)
                st.markdown("### ✏️ Manual Corrections")
                correction_col1, correction_col2 = st.columns([1, 1])
                with correction_col1:
                    correction_mode = st.radio("Correction Mode:", ["🟡 Yellow Pen (Healthy)", "⚫ Black Pen (Lesion)"])
                with correction_col2:
                    pen_size = st.slider("Pen Size:", 2, 15, 5)
                    if st.button("🔄 Recalculate"):
                        st.session_state.apply_corrections = True
                    if st.button("🗑️ Clear Corrections"):
                        st.session_state.canvas_key_counter += 1
                        st.session_state.corrected_result = None
                        st.info("Corrections cleared. You can start drawing again.")
                stroke_color = "rgba(255, 255, 0, 1)" if "Yellow" in correction_mode else "rgba(0, 0, 0, 1)"
                correction_canvas_key = f"trial_correction_canvas_{st.session_state.canvas_key_counter}"
                correction_canvas = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=pen_size,
                    stroke_color=stroke_color,
                    background_image=image,
                    update_streamlit=True,
                    height=image_np.shape[0],
                    width=image_np.shape[1],
                    drawing_mode="freedraw",
                    key=correction_canvas_key,
                )
                st.caption("Draw corrections and observe performance.")
                if st.session_state.apply_corrections:
                    st.session_state.apply_corrections = False
                    try:
                        if correction_canvas.image_data is not None:
                            correction_data = correction_canvas.image_data
                            yellow_correction = np.zeros((correction_data.shape[0], correction_data.shape[1]), dtype=np.uint8)
                            black_correction = np.zeros((correction_data.shape[0], correction_data.shape[1]), dtype=np.uint8)
                            yellow_pixels = ((correction_data[:,:,0] > 180) & (correction_data[:,:,1] > 180) & (correction_data[:,:,2] < 120) & (correction_data[:,:,3] > 50))
                            yellow_correction[yellow_pixels] = 255
                            black_pixels = ((correction_data[:,:,0] < 80) & (correction_data[:,:,1] < 80) & (correction_data[:,:,2] < 80) & (correction_data[:,:,3] > 50))
                            black_correction[black_pixels] = 255
                            if np.count_nonzero(yellow_correction) > 0 or np.count_nonzero(black_correction) > 0:
                                st.success(f"Yellow: {np.count_nonzero(yellow_correction)} | Black: {np.count_nonzero(black_correction)} pixels corrected.")
                            else:
                                st.warning("No corrections detected.")
                            corrected_lesion_mask = lesion_mask.copy()
                            corrected_total_mask = total_mask.copy()
                            corrected_lesion_mask[yellow_correction > 0] = 0
                            corrected_total_mask[yellow_correction > 0] = 255
                            corrected_lesion_mask[black_correction > 0] = 255
                            corrected_mango_area_px = np.count_nonzero(corrected_total_mask)
                            corrected_lesion_area_px = np.count_nonzero(corrected_lesion_mask)
                            if corrected_mango_area_px > 0:
                                mm_per_px_sq = st.session_state.mm_per_px * st.session_state.mm_per_px
                                corrected_mango_area_mm2 = corrected_mango_area_px * mm_per_px_sq
                                corrected_lesion_area_mm2 = corrected_lesion_area_px * mm_per_px_sq
                                corrected_lesion_percent = (corrected_lesion_area_mm2 / corrected_mango_area_mm2 * 100)
                                st.session_state.corrected_result = {
                                    "Sample": len(st.session_state.samples) + 1,
                                    "Area (mm²)": round(corrected_mango_area_mm2, 1),
                                    "Lesions (mm²)": round(corrected_lesion_area_mm2, 1),
                                    "Lesion %": round(corrected_lesion_percent, 1)
                                }
                                st.markdown("### 📊 Corrected Results")
                                corr_col1, corr_col2, corr_col3 = st.columns([1, 1, 1])
                                with corr_col1:
                                    st.image(corrected_total_mask, caption="🥭 Corrected Mango Area", width=120)
                                with corr_col2:
                                    st.image(corrected_lesion_mask, caption="🔴 Corrected Lesion Areas", width=120)
                                with corr_col3:
                                    st.metric("Total Area", f"{corrected_mango_area_mm2:.1f} mm²")
                                    st.metric("Lesion Area", f"{corrected_lesion_area_mm2:.1f} mm²")
                                    st.metric("Lesion %", f"{corrected_lesion_percent:.1f}%")
                            else:
                                st.warning("No mango detected after correction.")
                        else:
                            st.warning("No corrections detected.")
                    except Exception as e:
                        st.error(f"Correction error: {str(e)}")
                        st.session_state.corrected_result = None
                st.markdown("### 📋 Add Sample to Table")
                button_col1, button_col2 = st.columns(2)
                with button_col1:
                    if st.button("✅ Add Original Sample", key="add_original_btn"):
                        if len(st.session_state.samples) < TRIAL_MAX_SAMPLES:
                            st.session_state.samples.append(result)
                            st.success("Original sample added.")
                        else:
                            st.warning(f"Maximum samples reached ({TRIAL_MAX_SAMPLES}).")
                with button_col2:
                    if st.session_state.corrected_result is not None:
                        if st.button("✅ Add Corrected Sample", key="add_corrected_btn"):
                            if len(st.session_state.samples) < TRIAL_MAX_SAMPLES:
                                st.session_state.samples.append(st.session_state.corrected_result)
                                st.success("Corrected sample added.")
                            else:
                                st.warning(f"Maximum samples reached ({TRIAL_MAX_SAMPLES}).")
                    else:
                        st.button("⚪ Add Corrected Sample", disabled=True)
    if st.session_state.samples:
        st.markdown("### 📊 All Samples")
        df = pd.DataFrame(st.session_state.samples)
        st.dataframe(df, use_container_width=True)
        if len(st.session_state.samples) > 1:
            avg_lesion = df["Lesion %"].mean()
            max_lesion = df["Lesion %"].max()
            min_lesion = df["Lesion %"].min()
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.metric("Average Lesion %", f"{avg_lesion:.1f}%")
            with summary_col2:
                st.metric("Maximum Lesion %", f"{max_lesion:.1f}%")
            with summary_col3:
                st.metric("Minimum Lesion %", f"{min_lesion:.1f}%")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear All"):
                st.session_state.samples = []
                st.success("All samples cleared!")
        with col2:
            if st.session_state.samples:
                sample_to_delete = st.selectbox("🗑️ Delete specific sample:", options=[f"Sample {i}" for i in range(1, len(st.session_state.samples) + 1)])
                if st.button("Delete Selected"):
                    sample_idx = int(sample_to_delete.split()[1]) - 1
                    st.session_state.samples.pop(sample_idx)
                    st.success(f"Deleted: Sample {sample_idx + 1}")
        with col3:
            custom_filename = st.text_input("📁 Filename:", value="mango_trial_analysis")
            if custom_filename:
                safe_filename = "".join(c for c in custom_filename if c.isalnum() or c in "._-")
                if not safe_filename:
                    safe_filename = "mango_trial_analysis"
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, f"{safe_filename}.csv", "text/csv")
else:
    st.info("Upload a small mango image to begin trial.")
