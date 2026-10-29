"""
Pre-processing and Vision Utilities for Satellite Imagery
Handles:
1. Non-foliage (road/asphalt/built-up/water) contour detection using HSV mask.
2. Robust bounding crop around identified linear features.
3. Graceful fallback to smart central crop when mask conditions are unclear.
4. Optional debug visualization saving.
"""

import os
import cv2
import numpy as np
from PIL import Image

DEBUG_DIR = os.getenv("DEBUG_CROPS_DIR", "./debug_crops")


def ensure_debug_dir():
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR, exist_ok=True)


def auto_crop_linear_feature(
    pil_image: Image.Image,
    save_debug: bool = False,
    debug_filename: str = "debug_crop.jpg"
) -> Image.Image:
    """
    Extracts non-green linear structural features (e.g. roads, highways, rivers)
    from dominant foliage backgrounds using OpenCV.

    Pipeline:
    1. Convert PIL Image to HSV color space.
    2. Filter out dominant green canopy foliage (Hue ~ 35 to 85).
    3. Find external contours on the non-green binary mask.
    4. If a candidate contour meets minimum area criteria, compute its bounding box with padding.
    5. Fallback: If no distinct non-green feature is found, apply a 70% smart center crop.
    6. Optional: Save intermediate debug comparison image to disk.
    """
    try:
        # Convert PIL to OpenCV BGR format
        img_np = np.array(pil_image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape

        # Convert to HSV to separate chromaticity (Hue) from luminance
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Green foliage Hue range in OpenCV HSV (H is 0-179, S is 0-255, V is 0-255)
        # Trees, dense forest, crops, and grass typically reside between 32 and 88
        lower_green = np.array([30, 35, 30])
        upper_green = np.array([88, 255, 255])

        # Mask of all green canopy/vegetation
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Invert mask: Non-green areas (asphalt, concrete roads, waterways, buildings)
        non_green_mask = cv2.bitwise_not(green_mask)

        # Morphological noise removal: Remove tiny speckles and connect linear road segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(non_green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find external contours of non-green features
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = h * w
        # Filter contours by size: must be at least 1.5% of total image area to avoid sensor noise
        valid_contours = [c for c in contours if cv2.contourArea(c) > (total_area * 0.015)]

        crop_success = False
        crop_box = None

        if valid_contours:
            # Combine major contours into a unified bounding box
            all_pts = np.vstack(valid_contours)
            x, y, cw, ch = cv2.boundingRect(all_pts)

            # Apply proportional safety padding (15% padding)
            pad_x = int(cw * 0.15)
            pad_y = int(ch * 0.15)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)

            # Check that the cropped box is meaningful (not zero-sized and has reasonable area)
            if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                crop_box = (x1, y1, x2, y2)
                crop_success = True

        # Fallback: Smart Center Crop (70% central field of view)
        if not crop_success:
            crop_fraction = 0.70
            crop_w = int(w * crop_fraction)
            crop_h = int(h * crop_fraction)
            start_x = (w - crop_w) // 2
            start_y = (h - crop_h) // 2
            crop_box = (start_x, start_y, start_x + crop_w, start_y + crop_h)

        x1, y1, x2, y2 = crop_box
        cropped_np = img_np[y1:y2, x1:x2]

        # Optional Debug Visualization saving
        if save_debug:
            ensure_debug_dir()
            debug_path = os.path.join(DEBUG_DIR, debug_filename)
            # Create side-by-side inspection canvas: [Original with bounding box | Cropped Result]
            debug_vis = img_bgr.copy()
            box_color = (0, 255, 0) if crop_success else (0, 165, 255)  # Green if contour, Orange if center crop
            cv2.rectangle(debug_vis, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                debug_vis,
                "Auto-Contour" if crop_success else "Center-Crop Fallback",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1,
                cv2.LINE_AA
            )

            # Resize cropped for horizontal stacking preview
            resized_crop = cv2.resize(cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR), (w, h))
            combined_preview = np.hstack([debug_vis, resized_crop])
            cv2.imwrite(debug_path, combined_preview)
            print(f"📸 Debug image saved -> {debug_path} (Method: {'HSV-Contour' if crop_success else 'Fallback-Center'})")

        return Image.fromarray(cropped_np)

    except Exception as err:
        # Ultimate fail-safe: return original image if OpenCV encountering unparseable matrix
        print(f"⚠️ Preprocessor error encountered ({err}). Falling back to original image.")
        return pil_image
