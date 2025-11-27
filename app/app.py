import sys
from pathlib import Path
import os
import cv2
import numpy as np
import streamlit as st

# ============================================================
# FIX IMPORT PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.q1_stitch_panorama import crop_black_borders
from src.sift_from_scratch import SIFTFromScratch   # ONLY Q2 will use this


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CSC 8830 Assignment 4",
    layout="wide"
)

# ============================================================
# MAIN NAVIGATION (TABS ARE VISIBLE IMMEDIATELY)
# ============================================================
tab_q1, tab_q2 = st.tabs(["Image Stitching (Q1)", "SIFT + RANSAC (Q2)"])


# ============================================================
# ======================== Q1 PAGE ============================
# ============================================================
with tab_q1:

    st.title("Panorama Stitching Demonstration")

    st.markdown("### Original Dataset Images")

    pano_dir = ROOT / "data/panorama"
    phone_path = ROOT / "data/phone_panorama/phone_panorama.jpg"

    # Load fixed images
    fixed_paths = sorted(list(pano_dir.glob("*.jpg")))
    fixed_imgs = []

    if fixed_paths:
        cols = st.columns(5)
        for i, p in enumerate(fixed_paths):
            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fixed_imgs.append(img)

            with cols[i % 5]:
                st.image(img, caption=p.name, use_container_width=True)
    else:
        st.warning("No images found in data/panorama/")

    st.markdown("---")
    st.markdown("### Resulting Panorama and Phone Panorama")

    result_pano_path = ROOT / "output/q1_my_panorama.jpg"

    if result_pano_path.exists():
        pano = cv2.imread(str(result_pano_path))
        pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)

        phone = cv2.imread(str(phone_path))
        phone = cv2.cvtColor(phone, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(pano, caption="Your Stitched Panorama", use_container_width=True)
        with col2:
            st.image(phone, caption="Phone Panorama", use_container_width=True)
    else:
        st.info("Run q1_stitch_panorama.py to generate the stitched panorama.")

    st.markdown("---")
    st.markdown("### Generate a Panorama From Your Own Images")

    uploaded = st.file_uploader(
        "Upload at least 3 overlapping images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded and len(uploaded) >= 3:

        imgs = []
        for f in uploaded:
            data = np.frombuffer(f.read(), np.uint8)
            im = cv2.imdecode(data, cv2.IMREAD_COLOR)
            imgs.append(im)

        # OpenCV Stitcher
        try:
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        except AttributeError:
            stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

        status, pano_generated = stitcher.stitch(imgs)

        if status == cv2.Stitcher_OK:
            pano_generated = crop_black_borders(pano_generated)
            pano_rgb = cv2.cvtColor(pano_generated, cv2.COLOR_BGR2RGB)
            st.image(pano_rgb, caption="Generated Panorama", use_container_width=True)
        else:
            st.error(f"Stitching failed (error code {status}).")

    elif uploaded:
        st.warning("Please upload at least 3 images.")




# ============================================================
# ======================== Q2 PAGE ============================
# ============================================================
with tab_q2:

    st.title("SIFT From Scratch + RANSAC Demonstration")

    pair_dir = ROOT / "data/sift_pair"
    img1_path = pair_dir / "1.jpg"
    img2_path = pair_dir / "2.jpg"

    # Output paths
    out_my = ROOT / "output/q2_my_sift_matches.jpg"
    out_cv = ROOT / "output/q2_cv_sift_matches.jpg"

    # ---- Speed utility ----
    def downsample(img, max_dim=600):
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        return img

    # ========================================================
    # Show original pair immediately
    # ========================================================
    if not img1_path.exists() or not img2_path.exists():
        st.error("Missing images in data/sift_pair/")
    else:
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))

        st.markdown("### Original Image Pair")
        colA, colB = st.columns(2)
        with colA:
            st.image(img1[..., ::-1], caption="Image 1", use_container_width=True)
        with colB:
            st.image(img2[..., ::-1], caption="Image 2", use_container_width=True)

    st.markdown("---")

    # ========================================================
    # Show PRE-COMPUTED match visualizations if available
    # ========================================================
    st.markdown("### Match Visualizations (Precomputed)")

    col1, col2 = st.columns(2)

    with col1:
        if out_my.exists():
            im = cv2.imread(str(out_my))[..., ::-1]
            st.image(im, caption="My SIFT-from-Scratch Matches", use_container_width=True)
        else:
            st.info("Run the SIFT-from-scratch pipeline to generate this visualization.")

    with col2:
        if out_cv.exists():
            im = cv2.imread(str(out_cv))[..., ::-1]
            st.image(im, caption="OpenCV SIFT Matches", use_container_width=True)
        else:
            st.info("Run the SIFT-from-scratch pipeline to generate this visualization.")

    st.markdown("---")

    # ========================================================
    # Optional: Run SIFT-from-scratch & RANSAC manually
    # ========================================================
    st.header("Run SIFT-from-Scratch + RANSAC Manually")

    if st.button("Run SIFT-from-Scratch + RANSAC"):

        st.warning("Running SIFT-from-scratch… This may take several seconds.")

        # Load images fresh
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))

        # Downsample first (makes SIFT 50× faster)
        img1_small = downsample(img1)
        img2_small = downsample(img2)

        gray1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)

        sift = SIFTFromScratch()

        # ---- Detect features ----
        k1, d1 = sift.detectAndCompute(gray1)
        k2, d2 = sift.detectAndCompute(gray2)

        # Keypoint coordinates
        pts1 = np.array([[kp["x"], kp["y"]] for kp in k1], dtype=np.float32)
        pts2 = np.array([[kp["x"], kp["y"]] for kp in k2], dtype=np.float32)

        # ---- Ratio test ----
        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw = bf.knnMatch(d1, d2, k=2)
        matches = [m[0] for m in raw if len(m) == 2 and m[0].distance < 0.75*m[1].distance]

        # ---- RANSAC homography ----
        if len(matches) < 4:
            st.error("Not enough SIFT matches for RANSAC.")
        else:
            src = np.float32([pts1[m.queryIdx] for m in matches])
            dst = np.float32([pts2[m.trainIdx] for m in matches])

            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)

            # ---- Draw my matches ----
            canvas = np.zeros((max(img1_small.shape[0], img2_small.shape[0]),
                               img1_small.shape[1] + img2_small.shape[1], 3), dtype=np.uint8)
            canvas[0:img1_small.shape[0], 0:img1_small.shape[1]] = img1_small
            canvas[0:img2_small.shape[0], img1_small.shape[1]:] = img2_small

            for m, inl in zip(matches, mask.ravel()):
                if inl:
                    x1, y1 = src[m.queryIdx]
                    x2, y2 = dst[m.trainIdx]
                    cv2.circle(canvas, (int(x1), int(y1)), 3, (0,255,0), -1)
                    cv2.circle(canvas, (int(x2)+img1_small.shape[1], int(y2)), 3, (0,255,0), -1)
                    cv2.line(canvas, (int(x1), int(y1)),
                                     (int(x2)+img1_small.shape[1], int(y2)), (0,255,0), 1)

            cv2.imwrite(str(out_my), canvas)

            # ---- OpenCV SIFT for comparison ----
            cv_sift = cv2.SIFT_create()
            gray1_full = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2_full = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            kp1, des1 = cv_sift.detectAndCompute(gray1_full, None)
            kp2, des2 = cv_sift.detectAndCompute(gray2_full, None)

            bf2 = cv2.F
            bf2 = cv2.BFMatcher()
            raw2 = bf2.knnMatch(des1, des2, k=2)
            good2 = [m[0] for m in raw2 if len(m) == 2 and m[0].distance < 0.75*m[1].distance]

            if len(good2) >= 4:
                cv2.imwrite(str(out_cv),
                            cv2.drawMatches(img1, kp1, img2, kp2, good2, None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

        st.success("SIFT-from-scratch processing complete. Refresh the Q2 tab to see updated images.")
