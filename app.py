
import os
os.environ["STREAMLIT_SERVER_PORT"] = "80"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0:80"
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import gc
from io import BytesIO
import time
import concurrent.futures

# Cloud-optimized page config
st.set_page_config(
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_title="Mango Lesion Analyzer",
    page_icon="🥭"
)

st.title("🥭 Mango Lesion Analyzer")

# Streamlit Cloud optimized constants
MAX_IMAGE_SIZE = 250  # Further reduced for cloud
CANVAS_SIZE = 600     # Smaller canvas for better performance  
MAX_FILE_SIZE_MB = 4  # More conservative file size limit
MEMORY_CLEANUP_INTERVAL = 30  # More frequent cleanup
MAX_SAMPLES = 12      # Reduced sample limit

# Streamlit Cloud optimized constants - less aggressive, more stable
CANVAS_UPDATE_THROTTLE = 0.15   # Balanced throttle for transform mode (stability + speed)
TRANSFORM_DEBOUNCE_TIME = 0.13  # Balanced debounce for other modes
MAX_CANVAS_OPERATIONS = 18      # Middle ground for max operations before auto-reset
MAX_DISPLAY_DIM = 800           # More conservative for cloud memory
CANVAS_ERROR_COOLDOWN = 1.3     # Middle ground for error recovery cooldown
MAX_CONSECUTIVE_ERRORS = 4      # Slightly higher tolerance before auto-reset

# Initialize session state efficiently
session_defaults = {
    "samples": [],
    "mm_per_px": None,
    "polygon_drawing": False,
    "last_cleanup": time.time(),
    "last_cache_clear": 0,
    "canvas_key_counter": 0,
    "transform_mode_active": False,
    "last_canvas_update": 0,
    "transform_warning_shown": False,
    "canvas_operation_count": 0,
    "last_canvas_error": 0,
    "consecutive_canvas_errors": 0,  # New: track consecutive errors
    "canvas_disabled": False,        # New: temporary canvas disable flag
    "last_error_type": None,         # New: track error patterns
    "canvas_retry_count": 0          # New: track retry attempts
}

for key, default in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Periodic cleanup with memory monitoring - reduced frequency
current_time = time.time()
if current_time - st.session_state.get('last_cleanup', 0) > 60:  # Less frequent cleanup
    gc.collect()
    st.session_state.last_cleanup = current_time

def aggressive_cleanup():
    """Streamlit Cloud optimized cleanup - less disruptive"""
    try:
        gc.collect()
        
        # Only clear non-essential cache keys
        cleanup_keys = [
            'temp_canvas_data', 'temp_masks', 'large_arrays', 
            'temp_images', 'processed_images', 'analysis_results'
        ]
        for key in cleanup_keys:
            if key in st.session_state:
                try:
                    del st.session_state[key]
                except Exception:
                    pass
        
        # Less aggressive canvas key cleanup
        canvas_keys = [k for k in st.session_state.keys() if 'canvas' in k.lower()]
        current_counter = st.session_state.get('canvas_key_counter', 0)
        
        # Only remove very old canvas keys (keep last 3)
        for key in canvas_keys:
            if key != 'canvas_key_counter':
                try:
                    if any(old_key in key for old_key in ['mango_canvas_', 'correction_canvas_']):
                        # Keep last 3 canvas states
                        keep_keys = [f'mango_canvas_{current_counter}', 
                                   f'mango_canvas_{current_counter-1}',
                                   f'mango_canvas_{current_counter-2}']
                        if key not in keep_keys:
                            if key in st.session_state:
                                del st.session_state[key]
                except Exception:
                    pass
        
        st.session_state.last_cleanup = time.time()
        
        # Less frequent cache clearing - higher threshold
        try:
            if (time.time() - st.session_state.get('last_cache_clear', 0)) > MEMORY_CLEANUP_INTERVAL:
                # Only clear if really needed - higher threshold
                if len(st.session_state.keys()) > 50:
                    st.cache_data.clear()
                    st.session_state.last_cache_clear = time.time()
                    gc.collect()
        except Exception:
            pass
            
    except Exception as e:
        # Minimal fallback cleanup
        try:
            gc.collect()
            st.session_state.last_cleanup = time.time()
        except Exception:
            pass

def check_memory_limit():
    """Enhanced memory usage monitoring for cloud"""
    try:
        # Check samples limit
        if len(st.session_state.samples) > MAX_SAMPLES:
            st.error(f"💾 Maximum samples reached ({MAX_SAMPLES}). Please download and clear samples.")
            return True
        
        if len(st.session_state.samples) > MAX_SAMPLES * 0.6:  # 60% warning threshold
            st.warning("⚠️ Many samples stored. Consider downloading results soon.")
        
        # Monitor session state size - higher threshold
        session_keys = len(st.session_state.keys())
        if session_keys > 60:
            st.warning(f"⚠️ High memory usage detected ({session_keys} session keys). Auto-cleaning...")
            aggressive_cleanup()
            return False
            
        return False
        
    except Exception:
        return False

# Auto-cleanup session state if needed - higher threshold
if len(st.session_state.keys()) > 60:
    check_memory_limit()

def emergency_reset():
    """Emergency session state reset"""
    try:
        # Clear problematic keys
        problem_keys = ['polygon_drawing', 'temp_canvas_data', 'temp_masks', 'large_arrays', 'temp_images']
        for key in problem_keys:
            if key in st.session_state:
                try:
                    del st.session_state[key]
                except Exception:
                    pass
        
        # Clear canvas-related keys
        canvas_keys = [k for k in st.session_state.keys() if 'canvas' in k.lower()]
        for key in canvas_keys:
            if key in st.session_state:
                try:
                    del st.session_state[key]
                except Exception:
                    pass
        
        aggressive_cleanup()
        return True
    except Exception:
        return False

def safe_canvas_reset():
    """Streamlit Cloud optimized canvas reset - minimal disruption"""
    try:
        # Only reset essential canvas tracking
        st.session_state.canvas_operation_count = 0
        st.session_state.last_canvas_update = 0
        st.session_state.polygon_drawing = False
        st.session_state.transform_mode_active = False
        
        # Reset consecutive errors
        st.session_state.consecutive_canvas_errors = 0
        
        # Increment counter for new canvas key
        st.session_state.canvas_key_counter = st.session_state.get('canvas_key_counter', 0) + 1
        
        # Light cleanup
        gc.collect()
        
        return True
    except Exception:
        return False

def get_canvas_health_status():
    """Simplified canvas health status for better UX"""
    consecutive_errors = st.session_state.get('consecutive_canvas_errors', 0)
    op_count = st.session_state.get('canvas_operation_count', 0)
    
    if op_count > MAX_CANVAS_OPERATIONS * 0.9:
        return "� High Activity", "Auto-reset coming soon"
    elif consecutive_errors >= 3:
        return "� Some Issues", "Multiple recent errors"
    elif consecutive_errors > 0:
        return "� Minor Issues", "Recent errors detected"
    else:
        return "🟢 Canvas Ready", "Operating normally"

def handle_canvas_error(error_msg, error_type="general"):
    """Streamlit Cloud optimized error handling - less aggressive"""
    current_time = time.time()
    
    # Update error tracking more conservatively
    st.session_state.last_canvas_error = current_time
    st.session_state.last_error_type = error_type
    
    # Only increment errors for serious issues
    if any(keyword in error_msg.lower() for keyword in ["memory", "limit", "overload", "timeout"]):
        st.session_state.consecutive_canvas_errors = st.session_state.get('consecutive_canvas_errors', 0) + 1
    
    # Handle critical errors only
    if any(keyword in error_msg.lower() for keyword in ["memory", "overload"]):
        st.warning("💾 Memory pressure detected. Auto-optimizing...")
        aggressive_cleanup()
        return "memory_warning"
        
    elif any(keyword in error_msg.lower() for keyword in ["session", "initialized", "duplicate"]):
        # Just reset canvas, don't disable
        safe_canvas_reset()
        return "session_reset"
        
    elif "timeout" in error_msg.lower():
        st.info("⏳ Slow response detected. Optimizing...")
        return "timeout_info"
        
    else:
        # For most errors, just log and continue
        consecutive_errors = st.session_state.get('consecutive_canvas_errors', 0)
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            st.warning("⚠️ Multiple canvas issues. Resetting for stability...")
            safe_canvas_reset()
            st.session_state.consecutive_canvas_errors = 0
            return "reset_after_multiple"
        else:
            # Just show a brief warning and continue
            return "minor_error"

def is_canvas_stable():
    """Simplified canvas stability check for Streamlit Cloud"""
    current_time = time.time()
    
    # Only disable for very recent errors (shorter cooldown)
    last_error = st.session_state.get('last_canvas_error', 0)
    if current_time - last_error < CANVAS_ERROR_COOLDOWN:
        return False
    
    # Auto-reset only when operations are very high
    op_count = st.session_state.get('canvas_operation_count', 0)
    if op_count > MAX_CANVAS_OPERATIONS:
        safe_canvas_reset()
        st.session_state.canvas_operation_count = 0
        return True  # Allow immediate use after reset
    
    return True

@st.cache_data(max_entries=1, ttl=60, show_spinner=False)  # Reduced TTL for cloud
def process_uploaded_image(uploaded_file, max_dim=MAX_IMAGE_SIZE):
    """Cloud-optimized image processing with enhanced error handling and basic enhancement"""
    if not uploaded_file or len(uploaded_file) == 0:
        return None, None, None

    file_size_mb = len(uploaded_file) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large: {file_size_mb:.1f}MB. Use files under {MAX_FILE_SIZE_MB}MB.")
        return None, None, None

    try:
        image = Image.open(BytesIO(uploaded_file))
        original_size = image.size

        if original_size[0] * original_size[1] == 0:
            st.error("❌ Invalid image dimensions")
            return None, None, None

        total_pixels = original_size[0] * original_size[1]
        if total_pixels > 600000:
            st.warning("Large image detected. Optimizing for cloud.")
            max_dim = min(max_dim, 200)
        elif total_pixels > 400000:
            max_dim = min(max_dim, 220)

        if image.mode != 'RGB':
            image = image.convert("RGB")

        scale = min(max_dim / image.height, max_dim / image.width, 1.0)
        if scale < 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            st.info(f"Resized: {original_size[0]}x{original_size[1]} → {new_size[0]}x{new_size[1]}")

        # --- Enhancement: Lower shadows and increase saturation ---
        import cv2
        import numpy as np

        image_np = np.array(image, dtype=np.uint8)
        img_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

        # Reduce shadows: brighten dark pixels in V channel
        v = img_hsv[:,:,2]
        shadow_mask = v < 80
        v[shadow_mask] = np.clip(v[shadow_mask] + 40, 0, 255)
        img_hsv[:,:,2] = v

        # Increase saturation
        s = img_hsv[:,:,1]
        s = np.clip(s * 1.25, 0, 255)
        img_hsv[:,:,1] = s.astype(np.uint8)

        image_np = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        del image
        gc.collect()

        return image_np, original_size, scale

    except MemoryError:
        st.error("❌ Memory limit exceeded. Please use a smaller image.")
        gc.collect()
        return None, None, None
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        gc.collect()
        return None, None, None



def quick_color_analysis(image_np, mask, mm_per_px):
    """Cloud-optimized color analysis with memory management"""
    if image_np is None or mask is None or mm_per_px is None:
        return 0, 0, 0, None, None
    
    if mask.size == 0 or np.max(mask) == 0:
        return 0, 0, 0, None, None
    
    try:
        if mask.shape != image_np.shape[:2]:
            mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # HSV conversion with memory optimization
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Optimize mask operations by combining them
        # Healthy mango detection - combine operations for efficiency
        mango_mask1 = cv2.inRange(hsv, (20, 40, 60), (85, 255, 255))
        mango_mask2 = cv2.inRange(hsv, (15, 25, 40), (80, 200, 200))
        mango_mask3 = cv2.inRange(hsv, (25, 20, 100), (75, 150, 255))
        
        # Use in-place operations where possible
        healthy_mask = cv2.bitwise_or(mango_mask1, mango_mask2)
        cv2.bitwise_or(healthy_mask, mango_mask3, healthy_mask)  # In-place operation
        
        # Clear intermediate masks to save memory
        del mango_mask1, mango_mask2, mango_mask3
        
        # Lesion detection - optimized
        lesion_mask1 = cv2.inRange(hsv, (8, 50, 30), (25, 255, 140))
        lesion_mask2 = cv2.inRange(hsv, (0, 60, 25), (15, 255, 120))
        lesion_mask3 = cv2.inRange(hsv, (0, 0, 0), (180, 255, 45))
        lesion_mask4 = cv2.inRange(hsv, (0, 10, 20), (180, 80, 100))
        
        # Combine lesion masks efficiently
        raw_lesion_mask = cv2.bitwise_or(lesion_mask1, lesion_mask2)
        cv2.bitwise_or(raw_lesion_mask, lesion_mask3, raw_lesion_mask)
        cv2.bitwise_or(raw_lesion_mask, lesion_mask4, raw_lesion_mask)
        
        # Clear intermediate masks
        del lesion_mask1, lesion_mask2, lesion_mask3, lesion_mask4, hsv
        
        # Apply user mask in-place
        cv2.bitwise_and(healthy_mask, mask, healthy_mask)
        cv2.bitwise_and(raw_lesion_mask, mask, raw_lesion_mask)
        
        final_lesion_mask = raw_lesion_mask  # No morphology operations
        total_mango_mask = cv2.bitwise_or(healthy_mask, final_lesion_mask)
        
        # Calculate areas efficiently
        mango_area_px = np.count_nonzero(total_mango_mask)
        lesion_area_px = np.count_nonzero(final_lesion_mask)
        
        if mango_area_px == 0:
            return 0, 0, 0, None, None
        
        # Efficient area calculations
        mm_per_px_sq = mm_per_px * mm_per_px
        mango_area_mm2 = mango_area_px * mm_per_px_sq
        lesion_area_mm2 = lesion_area_px * mm_per_px_sq
        lesion_percent = (lesion_area_mm2 / mango_area_mm2 * 100) if mango_area_mm2 > 0 else 0
        
        # Force garbage collection for large operations
        gc.collect()
        
        return mango_area_mm2, lesion_area_mm2, lesion_percent, total_mango_mask, final_lesion_mask
        
    except MemoryError:
        st.error("❌ Memory limit reached during analysis. Try a smaller image.")
        gc.collect()
        return 0, 0, 0, None, None
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        gc.collect()
        return 0, 0, 0, None, None

def safe_rerun():
    """Safe app rerun with fallbacks"""
    aggressive_cleanup()
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            # Safe session state cleanup
            keys_to_preserve = ['samples', 'mm_per_px', 'last_cleanup']
            keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_preserve]
            
            for key in keys_to_delete:
                if key in st.session_state:
                    try:
                        del st.session_state[key]
                    except Exception:
                        pass
            
            st.info("🔄 App optimized. Continue with your analysis.")

def safe_add_sample(result):
    """Safely add sample with validation"""
    try:
        if 'samples' not in st.session_state:
            st.session_state.samples = []
        
        if len(st.session_state.samples) >= MAX_SAMPLES:
            st.warning(f"⚠️ Maximum samples reached ({MAX_SAMPLES}). Clear some samples before adding more.")
            return False
        
        if not result or not isinstance(result, dict):
            st.error("❌ Invalid result data")
            return False
        
        required_keys = ["Sample", "Area (mm²)", "Lesions (mm²)", "Lesion %"]
        if not all(key in result for key in required_keys):
            st.error("❌ Missing required result fields")
            return False
        
        # Validate numeric values
        try:
            area = float(result["Area (mm²)"])
            lesions = float(result["Lesions (mm²)"])
            percent = float(result["Lesion %"])
            
            if area < 0 or lesions < 0 or percent < 0:
                st.error("❌ Invalid negative values")
                return False
        except (ValueError, TypeError):
            st.error("❌ Invalid numeric values")
            return False
        
        st.session_state.samples.append(result)
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to add sample: {str(e)}")
        return False

# File uploader
uploaded_file = st.file_uploader(
    "Upload mango image (JPG/PNG)", 
    type=["png", "jpg", "jpeg"],
    help=f"Max size: {MAX_FILE_SIZE_MB}MB"
)

# Threaded execution helper
def run_in_thread(func, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result()

if uploaded_file:
    try:
        # Memory check
        if check_memory_limit():
            st.error("❌ Memory limit reached. Please try a smaller image or clear samples.")
            st.stop()
        
        # File size validation
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size > MAX_FILE_SIZE_MB:
            st.error(f"📁 File too large: {file_size:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_MB}MB")
            st.stop()
        
        # File size feedback
        if file_size > 4:
            st.warning(f"⚠️ Large file: {file_size:.1f}MB - expect slower cloud processing")
        elif file_size > 2:
            st.info(f"📏 File size: {file_size:.1f}MB")
        else:
            st.success(f"✅ File size: {file_size:.1f}MB")
        
        # Process image in thread
        with st.spinner("🔄 Processing image..."):
            image_np, original_size, scale = run_in_thread(process_uploaded_image, uploaded_file.getvalue())
        
        if image_np is None:
            st.error("❌ Failed to process image. Try a smaller file or different format.")
            st.stop()
        
        # Store original for analysis
        original_image_np = image_np.copy()
        h, w = image_np.shape[:2]
        st.success(f"✅ Image loaded: {w}x{h} pixels (scale: {scale:.2f})")
        
        # Calculate display size
        display_scale = min(CANVAS_SIZE / w, CANVAS_SIZE / h, 1.0)
        display_w = int(w * display_scale)
        display_h = int(h * display_scale)
        
        # Ensure minimum usable size
        MIN_CANVAS_SIZE = 300
        if display_w < MIN_CANVAS_SIZE or display_h < MIN_CANVAS_SIZE:
            min_scale = max(MIN_CANVAS_SIZE / w, MIN_CANVAS_SIZE / h)
            if min_scale <= 3.0:
                display_w = int(w * min_scale)
                display_h = int(h * min_scale)
                display_scale = min_scale
                st.info(f"🔧 Canvas enlarged for better usability: {display_w}x{display_h}")
        
        # Cloud optimization limits - more aggressive
        if display_w > MAX_DISPLAY_DIM or display_h > MAX_DISPLAY_DIM:
            scale_factor = min(MAX_DISPLAY_DIM / display_w, MAX_DISPLAY_DIM / display_h)
            display_w = int(display_w * scale_factor)
            display_h = int(display_h * scale_factor)
            display_scale *= scale_factor
            st.info(f"🔧 Canvas optimized for cloud: {display_w}x{display_h}")
        
        # Create display image with memory optimization
        try:
            display_image = Image.fromarray(image_np)
            if display_scale != 1.0:
                display_image = display_image.resize((display_w, display_h), Image.Resampling.LANCZOS)
        except MemoryError:
            st.error("❌ Memory limit reached creating display. Please use a smaller image.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Display creation failed: {str(e)}")
            st.stop()
        
        # Step 1: Scale Setting
        st.markdown("## 1️⃣ Set Scale")
        st.info("📏 Draw a line on a known measurement (ruler/scale bar)")
        st.success(f"✅ Canvas size: {display_w}x{display_h} pixels")
        
        scale_canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=3,
            stroke_color="rgba(255,0,0,1)",
            background_image=display_image,
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="line",
            key="scale_canvas",
        )
        
        scale_length_mm = st.number_input(
            "Real length of drawn line (mm):",
            min_value=0.1,
            value=10.0,
            step=1.0
        )
        
        # Calculate scale
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
                            if display_scale != 1.0:
                                scale_px /= display_scale
                            st.success(f"📏 Line drawn: {scale_px:.1f} pixels")
                        else:
                            scale_px = None
                    except (KeyError, ValueError, TypeError):
                        scale_px = None
        
        if scale_px and scale_length_mm > 0:
            st.session_state.mm_per_px = scale_length_mm / scale_px
            st.success(f"✅ Scale set: {st.session_state.mm_per_px:.4f} mm/pixel")
            
            # Step 2: Mango Analysis
            st.markdown("## 2️⃣ Analyze Mango")
            st.info("🥭 Draw around one mango at a time for accurate analysis")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                drawing_mode = st.radio(
                    "Drawing mode:",
                    ["circle", "rect", "polygon", "transform"],
                    horizontal=True
                )
                
                # Handle transform mode carefully
                if drawing_mode == "transform":
                    if not st.session_state.get("transform_mode_active", False):
                        st.session_state.transform_mode_active = True
                        # Show cloud-specific warning for transform mode
                        if not st.session_state.get("transform_warning_shown", False):
                            st.warning("⚠️ **Cloud Performance Tip**: Transform mode can be resource-intensive. For best results:\n"
                                     "• Drag shapes **slowly** to avoid crashes\n"
                                     "• Use other drawing modes (circle/rect) for faster performance\n"
                                     "• If app becomes unresponsive, click 'Reset Canvas'")
                            st.session_state.transform_warning_shown = True
                        st.info("🔄 Transform mode: Click existing shapes to modify them. **Drag slowly** to prevent cloud overload.")
                elif st.session_state.get("transform_mode_active", False):
                    st.session_state.transform_mode_active = False
                    safe_canvas_reset()
            
            with col2:
                brightness = st.slider("🔆 Brightness:", 0.7, 1.5, 1.0, 0.1)
                
                # Canvas health status
                health_status, health_desc = get_canvas_health_status()
                st.caption(f"{health_status}: {health_desc}")
                
                # Cloud-optimized controls
                if st.button("🔄 Reset Canvas", help="Clear canvas and fix any issues"):
                    safe_canvas_reset()
                    st.success("✅ Canvas reset completed")
                
                # Emergency stability button
                if st.button("🚨 Emergency Reset", help="Use if app becomes unresponsive", type="secondary"):
                    if emergency_reset():
                        st.success("✅ Emergency reset completed")
                        safe_rerun()
                
                # Performance monitoring for transform mode
                if drawing_mode == "transform":
                    op_count = st.session_state.get('canvas_operation_count', 0)
                    if op_count > 0:
                        st.caption(f"💡 Operations: {op_count}/{MAX_CANVAS_OPERATIONS}")
                    st.caption("⚠️ **Cloud Tip**: Drag shapes slowly to prevent crashes")

            # Apply brightness adjustment with memory management
            if brightness != 1.0:
                try:
                    # Use more memory-efficient brightness adjustment
                    display_array = np.array(display_image, dtype=np.float32)
                    np.multiply(display_array, brightness, out=display_array)  # In-place operation
                    np.clip(display_array, 0, 255, out=display_array)  # In-place clipping
                    adjusted_display_image = Image.fromarray(display_array.astype(np.uint8))
                    del display_array  # Clean up immediately
                except MemoryError:
                    st.warning("⚠️ Memory limit reached for brightness adjustment. Using original brightness.")
                    adjusted_display_image = display_image
                except Exception:
                    adjusted_display_image = display_image
            else:
                adjusted_display_image = display_image
            
            # Mode instructions with cloud-specific tips
            mode_instructions = {
                "transform": "🔄 **Transform Mode**: Click and drag shapes to move, drag corners/edges to resize. ⚠️ **Cloud users**: Drag slowly to prevent crashes!",
                "polygon": "📐 **Polygon Mode**: Click to place points, double-click to close polygon. ✅ **Cloud-friendly**",
                "circle": "✏️ **Circle Mode**: Draw a new circle around the mango. ✅ **Cloud-optimized**",
                "rect": "✏️ **Rectangle Mode**: Draw a new rectangle around the mango. ✅ **Cloud-optimized**"
            }
            instruction_text = mode_instructions.get(drawing_mode, "")
            if drawing_mode == "transform":
                st.warning(instruction_text)
            else:
                st.info(instruction_text)
            
            # Main analysis canvas with enhanced stability controls
            canvas_key = f"mango_canvas_{st.session_state.get('canvas_key_counter', 0)}"
            
            # Simplified cloud stability checks
            current_time = time.time()
            canvas_stable = is_canvas_stable()
            
            if not canvas_stable:
                st.info("� Canvas optimizing... Please wait a moment.")
                st.stop()
            
            # Simplified update throttling
            can_update_canvas = True
            last_update = st.session_state.get('last_canvas_update', 0)
            
            if drawing_mode == "transform":
                # Light throttling for transform mode
                if current_time - last_update < CANVAS_UPDATE_THROTTLE:
                    can_update_canvas = False
                else:
                    st.session_state.last_canvas_update = current_time
                    st.session_state.canvas_operation_count = st.session_state.get('canvas_operation_count', 0) + 1
                    
                    # Cleanup every 15 operations for stability
                    if st.session_state.canvas_operation_count % 15 == 0:
                        aggressive_cleanup()
            else:
                # Minimal throttling for other modes
                if current_time - last_update < TRANSFORM_DEBOUNCE_TIME:
                    can_update_canvas = False
                else:
                    st.session_state.last_canvas_update = current_time
            
            try:
                # Simplified canvas rendering for Streamlit Cloud
                if can_update_canvas:
                    canvas_result = st_canvas(
                        fill_color="rgba(255,165,0,0.2)",
                        stroke_width=3,
                        stroke_color="rgba(255,165,0,1)",
                        background_image=adjusted_display_image,
                        update_streamlit=True,
                        height=display_h,
                        width=display_w,
                        drawing_mode=drawing_mode,
                        key=canvas_key,
                    )
                    
                    # Reset error count on successful creation
                    if canvas_result is not None:
                        st.session_state.consecutive_canvas_errors = 0
                        
                else:
                    # Static canvas when throttled
                    canvas_result = st_canvas(
                        fill_color="rgba(255,165,0,0.2)",
                        stroke_width=3,
                        stroke_color="rgba(255,165,0,1)",
                        background_image=adjusted_display_image,
                        update_streamlit=False,
                        height=display_h,
                        width=display_w,
                        drawing_mode=drawing_mode,
                        key=canvas_key,
                    )
                    
                    # Show brief throttling message
                    if drawing_mode == "transform":
                        st.caption("⏳ Optimizing canvas performance...")
                
                # Simplified status display
                if drawing_mode == "transform":
                    op_count = st.session_state.get('canvas_operation_count', 0)
                    if op_count > MAX_CANVAS_OPERATIONS * 0.8:
                        st.caption(f"⚠️ High activity: {op_count}/{MAX_CANVAS_OPERATIONS}")
                    elif not can_update_canvas:
                        st.caption("⏳ Throttling active for stability")
                
            except Exception as e:
                error_msg = str(e)
                error_type = handle_canvas_error(error_msg)
                canvas_result = None
                
                # Simplified error recovery advice
                if error_type == "memory_warning":
                    st.info("💡 Try using a smaller image or circle/rect modes")
                elif error_type == "session_reset":
                    st.info("💡 Canvas reset - you can continue drawing")
                elif error_type == "reset_after_multiple":
                    st.info("💡 Canvas optimized - try circle/rect modes for better stability")
                elif error_type == "minor_error":
                    # Only show error in debug mode
                    pass

            # Process analysis
            process_analysis = False
            if (canvas_result and hasattr(canvas_result, 'image_data') and 
                canvas_result.image_data is not None):
                
                if (len(canvas_result.image_data.shape) >= 3 and 
                    canvas_result.image_data.shape[2] >= 4):
                    
                    alpha_channel = canvas_result.image_data[:,:,3]
                    has_drawing = np.any(alpha_channel > 0)
                    
                    if has_drawing:
                        if drawing_mode == "polygon":
                            if (canvas_result.json_data and 
                                canvas_result.json_data.get("objects")):
                                objects = canvas_result.json_data["objects"]
                                polygon_objects = [obj for obj in objects if obj and obj.get("type") == "polygon"]
                                
                                if polygon_objects:
                                    if not st.session_state.get("polygon_drawing", False):
                                        st.success("✅ Polygon completed! Computing analysis...")
                                        st.session_state.polygon_drawing = True
                                    process_analysis = True
                                else:
                                    st.info("🔄 Drawing polygon... Double-click to close")
                                    st.session_state.polygon_drawing = False
                        else:
                            process_analysis = True
            
            if process_analysis:
                try:
                    image_data = canvas_result.image_data
                    alpha_channel = image_data[:,:,3]
                    mask = (alpha_channel > 0).astype(np.uint8) * 255
                    
                    if np.max(mask) == 0:
                        st.warning("⚠️ No selection detected. Please draw on the image.")
                        st.stop()
                    
                    # Scale mask to match processing image
                    if display_scale != 1.0:
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Analyze with original image in thread
                    with st.spinner("🔬 Analyzing mango..."):
                        mango_area_mm2, lesion_area_mm2, lesion_percent, total_mask, lesion_mask = run_in_thread(
                            quick_color_analysis, original_image_np, mask, st.session_state.mm_per_px
                        )
                    
                    if mango_area_mm2 > 0:
                        # Display results
                        st.markdown("### 📊 Analysis Results")
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            st.image(total_mask, caption="🥭 Total Mango Area", width=180)
                        with col2:
                            st.image(lesion_mask, caption="🔴 Lesion Areas", width=180)
                        with col3:
                            st.metric("Total Area", f"{mango_area_mm2:.1f} mm²")
                            st.metric("Lesion Area", f"{lesion_area_mm2:.1f} mm²")
                            st.metric("Lesion %", f"{lesion_percent:.1f}%")
                        
                        # Results table
                        result = {
                            "Sample": len(st.session_state.samples) + 1,
                            "Area (mm²)": round(mango_area_mm2, 1),
                            "Lesions (mm²)": round(lesion_area_mm2, 1),
                            "Lesion %": round(lesion_percent, 1)
                        }
                        
                        st.dataframe(pd.DataFrame([result]), use_container_width=True)
                        
                        # Manual correction tools
                        st.markdown("### ✏️ Manual Corrections")
                        st.info("🎨 Use colored pens to correct misclassified areas. You can switch between colors and draw multiple corrections on the same canvas.")
                        
                        correction_col1, correction_col2 = st.columns([1, 1])
                        
                        with correction_col1:
                            correction_mode = st.radio(
                                "Correction Mode:",
                                ["🟡 Yellow Pen (Mark as Healthy)", "⚫ Black Pen (Mark as Lesion)"],
                                key="correction_mode",
                                help="Switch between modes to apply different corrections. Previously drawn marks will be preserved."
                            )
                        
                        with correction_col2:
                            pen_size = st.slider("Pen Size:", 2, 15, 5, key="pen_size")
                            if st.button("🔄 Recalculate"):
                                st.session_state.apply_corrections = True
                            if st.button("🗑️ Clear Corrections", help="Clear all yellow and black markings"):
                                # Reset the correction canvas by incrementing counter
                                st.session_state.canvas_key_counter = st.session_state.get('canvas_key_counter', 0) + 1
                                st.session_state.corrected_result = None
                                st.info("✅ Corrections cleared. You can start drawing again.")
                        
                        # Set stroke color
                        stroke_color = "rgba(255, 255, 0, 1)" if "Yellow" in correction_mode else "rgba(0, 0, 0, 1)"
                        
                        # Create overlay image with optimized memory usage
                        try:
                            # Avoid creating large intermediate arrays
                            overlay_image = np.array(adjusted_display_image, dtype=np.uint8)
                            
                            if display_scale != 1.0:
                                # Use more memory-efficient resizing
                                display_total_mask = cv2.resize(total_mask, (display_w, display_h), 
                                                              interpolation=cv2.INTER_NEAREST)
                                display_lesion_mask = cv2.resize(lesion_mask, (display_w, display_h), 
                                                               interpolation=cv2.INTER_NEAREST)
                            else:
                                display_total_mask = total_mask
                                display_lesion_mask = lesion_mask

                            # Optimize overlay operations
                            healthy_overlay = display_total_mask > 0
                            lesion_overlay = display_lesion_mask > 0
                            
                            # In-place operations for memory efficiency
                            healthy_only = healthy_overlay & ~lesion_overlay
                            if np.any(healthy_only):
                                overlay_image[healthy_only] = (overlay_image[healthy_only].astype(np.float32) * 0.7 + 
                                                              np.array([0, 100, 0], dtype=np.float32) * 0.3).astype(np.uint8)
                            
                            if np.any(lesion_overlay):
                                overlay_image[lesion_overlay] = (overlay_image[lesion_overlay].astype(np.float32) * 0.7 + 
                                                               np.array([100, 0, 0], dtype=np.float32) * 0.3).astype(np.uint8)

                            overlay_pil = Image.fromarray(overlay_image)
                            
                            # Clean up intermediate arrays
                            del overlay_image, healthy_overlay, lesion_overlay
                            if display_scale != 1.0:
                                del display_total_mask, display_lesion_mask
                                
                        except MemoryError:
                            st.warning("⚠️ Memory limit reached for overlay. Using simplified view.")
                            overlay_pil = adjusted_display_image
                        except Exception:
                            overlay_pil = adjusted_display_image

                        # Correction canvas - use stable key to preserve markings
                        correction_canvas_key = f"correction_canvas_{st.session_state.get('canvas_key_counter', 0)}"
                        correction_canvas = st_canvas(
                            fill_color="rgba(0,0,0,0)",
                            stroke_width=pen_size,
                            stroke_color=stroke_color,
                            background_image=overlay_pil,
                            update_streamlit=True,
                            height=display_h,
                            width=display_w,
                            drawing_mode="freedraw",
                            key=correction_canvas_key,
                        )

                        st.caption("🟢 Green tint = Healthy areas | 🔴 Red tint = Detected lesions")
                        st.caption("💡 **Tip**: Switch between Yellow and Black pen to make different corrections on the same canvas")
                        
                        # Initialize corrected result storage
                        if 'corrected_result' not in st.session_state:
                            st.session_state.corrected_result = None
                        
                        # Apply corrections
                        if st.session_state.get('apply_corrections', False):
                            st.session_state.apply_corrections = False
                            
                            try:
                                if correction_canvas.image_data is not None:
                                    correction_data = correction_canvas.image_data
                                    
                                    # Extract correction masks for both colors
                                    yellow_correction = np.zeros((correction_data.shape[0], correction_data.shape[1]), dtype=np.uint8)
                                    black_correction = np.zeros((correction_data.shape[0], correction_data.shape[1]), dtype=np.uint8)
                                    
                                    # Detect yellow strokes (more tolerant thresholds)
                                    yellow_pixels = ((correction_data[:,:,0] > 180) & 
                                                   (correction_data[:,:,1] > 180) & 
                                                   (correction_data[:,:,2] < 120) & 
                                                   (correction_data[:,:,3] > 50))
                                    yellow_correction[yellow_pixels] = 255
                                    
                                    # Detect black strokes (more tolerant thresholds)
                                    black_pixels = ((correction_data[:,:,0] < 80) & 
                                                  (correction_data[:,:,1] < 80) & 
                                                  (correction_data[:,:,2] < 80) & 
                                                  (correction_data[:,:,3] > 50))
                                    black_correction[black_pixels] = 255
                                    
                                    # Show detection feedback
                                    yellow_count = np.count_nonzero(yellow_correction)
                                    black_count = np.count_nonzero(black_correction)
                                    
                                    if yellow_count > 0 or black_count > 0:
                                        feedback_col1, feedback_col2 = st.columns(2)
                                        with feedback_col1:
                                            if yellow_count > 0:
                                                st.success(f"🟡 Yellow corrections: {yellow_count} pixels")
                                        with feedback_col2:
                                            if black_count > 0:
                                                st.success(f"⚫ Black corrections: {black_count} pixels")
                                    else:
                                        st.warning("⚠️ No corrections detected. Make sure to draw on the canvas before recalculating.")
                                    
                                    # Scale corrections
                                    if display_scale != 1.0:
                                        yellow_correction = cv2.resize(yellow_correction, (w, h), interpolation=cv2.INTER_NEAREST)
                                        black_correction = cv2.resize(black_correction, (w, h), interpolation=cv2.INTER_NEAREST)
                                    
                                    # Apply corrections
                                    corrected_lesion_mask = lesion_mask.copy()
                                    corrected_total_mask = total_mask.copy()
                                    
                                    # Yellow pen: remove from lesions, add to healthy
                                    corrected_lesion_mask[yellow_correction > 0] = 0
                                    corrected_total_mask[yellow_correction > 0] = 255
                                    
                                    # Black pen: add to lesions
                                    corrected_lesion_mask[black_correction > 0] = 255
                                    
                                    # Recalculate areas
                                    corrected_mango_area_px = np.count_nonzero(corrected_total_mask)
                                    corrected_lesion_area_px = np.count_nonzero(corrected_lesion_mask)
                                    
                                    if corrected_mango_area_px > 0:
                                        mm_per_px_sq = st.session_state.mm_per_px * st.session_state.mm_per_px
                                        corrected_mango_area_mm2 = corrected_mango_area_px * mm_per_px_sq
                                        corrected_lesion_area_mm2 = corrected_lesion_area_px * mm_per_px_sq
                                        corrected_lesion_percent = (corrected_lesion_area_mm2 / corrected_mango_area_mm2 * 100)
                                        
                                        # Store corrected result in session state
                                        st.session_state.corrected_result = {
                                            "Sample": len(st.session_state.samples) + 1,
                                            "Area (mm²)": round(corrected_mango_area_mm2, 1),
                                            "Lesions (mm²)": round(corrected_lesion_area_mm2, 1),
                                            "Lesion %": round(corrected_lesion_percent, 1)
                                        }
                                        
                                        # Display corrected results
                                        st.markdown("### 📊 Corrected Results")
                                        corr_col1, corr_col2, corr_col3 = st.columns([1, 1, 1])
                                        
                                        with corr_col1:
                                            st.image(corrected_total_mask, caption="🥭 Corrected Total Area", width=180)
                                        with corr_col2:
                                            st.image(corrected_lesion_mask, caption="🔴 Corrected Lesions", width=180)
                                        with corr_col3:
                                            area_change = corrected_mango_area_mm2 - mango_area_mm2
                                            lesion_change = corrected_lesion_percent - lesion_percent
                                            
                                            st.metric("Corrected Area", f"{corrected_mango_area_mm2:.1f} mm²", delta=f"{area_change:+.1f} mm²")
                                            st.metric("Corrected Lesions", f"{corrected_lesion_area_mm2:.1f} mm²")
                                            st.metric("Corrected %", f"{corrected_lesion_percent:.1f}%", delta=f"{lesion_change:+.1f}%")
                                        
                                        # Show corrected results table
                                        st.dataframe(pd.DataFrame([st.session_state.corrected_result]), use_container_width=True)
                                        st.success("✅ Corrections applied!")
                                    else:
                                        st.session_state.corrected_result = None
                                        st.error("❌ Corrected area calculation failed")
                                else:
                                    st.session_state.corrected_result = None
                                    st.warning("⚠️ No corrections detected")
                            except Exception as e:
                                st.error(f"❌ Correction error: {str(e)}")
                                st.session_state.corrected_result = None
                        
                        # Add Sample Buttons Section
                        st.markdown("### 📋 Add Sample to Table")
                        button_col1, button_col2 = st.columns(2)
                        
                        with button_col1:
                            if st.button("✅ Add Original Sample", key="add_original_btn", type="primary", use_container_width=True):
                                if safe_add_sample(result):
                                    st.success(f"✅ Sample {len(st.session_state.samples)} added (original values)!")
                                    # ...existing code...
                        
                        with button_col2:
                            if st.session_state.corrected_result is not None:
                                if st.button("✅ Add Corrected Sample", key="add_corrected_btn", type="primary", use_container_width=True):
                                    if safe_add_sample(st.session_state.corrected_result):
                                        st.success(f"✅ Sample {len(st.session_state.samples)} added (corrected values)!")
                                    # ...existing code...
                            else:
                                st.button("⚪ Add Corrected Sample", disabled=True, use_container_width=True, help="Apply corrections first by clicking 'Recalculate'")
                    else:
                        st.warning("⚠️ No mango detected. Try adjusting your selection or brightness.")
                        
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")
                    aggressive_cleanup()
            
            # Sample management
            if st.session_state.samples:
                st.markdown("### 📊 All Samples")
                df = pd.DataFrame(st.session_state.samples)
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
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
                
                # Management controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🗑️ Clear All", use_container_width=True):
                        st.session_state.samples = []
                        aggressive_cleanup()
                        st.success("✅ All samples cleared!")
                        safe_rerun()
                        
                with col2:
                    if st.session_state.samples:
                        sample_to_delete = st.selectbox(
                            "🗑️ Delete specific sample:",
                            options=[f"Sample {i}" for i in range(1, len(st.session_state.samples) + 1)],
                            key="delete_sample_select"
                        )
                        if st.button("Delete Selected", use_container_width=True):
                            sample_idx = int(sample_to_delete.split()[1]) - 1
                            st.session_state.samples.pop(sample_idx)
                            # Renumber remaining samples
                            for i, sample in enumerate(st.session_state.samples):
                                sample["Sample"] = i + 1
                            st.success(f"✅ Deleted: Sample {sample_idx + 1}")
                            aggressive_cleanup()
                            safe_rerun()
                
                with col3:
                    custom_filename = st.text_input(
                        "📁 Filename:",
                        value="mango_analysis",
                        help="Enter filename for CSV export",
                        key="custom_filename"
                    )
                    
                    if custom_filename:
                        safe_filename = "".join(c for c in custom_filename if c.isalnum() or c in "._-")
                        if not safe_filename:
                            safe_filename = "mango_analysis"
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            f"{safe_filename}.csv",
                            "text/csv",
                            use_container_width=True
                        )
        else:
            st.info("👆 **Step 1:** Draw a line on the scale bar and set its real length to continue")
            
    except MemoryError:
        aggressive_cleanup()
        st.error("❌ Cloud memory limit reached!")
        if st.button("🧹 Clear Memory"):
            keys_to_preserve = ['samples']
            keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_preserve]
            
            for key in keys_to_delete:
                if key in st.session_state:
                    try:
                        del st.session_state[key]
                    except Exception:
                        pass
            
            aggressive_cleanup()
            safe_rerun()

    except Exception as e:
        aggressive_cleanup()
        st.error(f"❌ Processing error: {str(e)}")
        if st.button("🔄 Restart Application"):
            safe_rerun()

# Sidebar controls
with st.sidebar:
    st.markdown("### 🔧 Quick Actions")
    if st.button("🔄 Reset App"):
        if emergency_reset():
            try:
                st.rerun()
            except Exception:
                st.info("App reset. Please refresh page if needed.")
    
    if st.button("🧹 Clear Memory"):
        aggressive_cleanup()
        st.success("✅ Memory cleared")

