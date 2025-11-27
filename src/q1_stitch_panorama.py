#!/usr/bin/env python3
"""
q1_stitch_panorama.py
---------------------

CSC 8830 – Computer Vision
Assignment 4 – Question 1

This script:
1. Loads multiple overlapping images taken by the student.
2. Uses OpenCV’s Stitcher API to automatically:
   - detect features
   - match keypoints
   - estimate camera transforms
   - warp images
   - blend them into a panorama
3. Removes black borders from the panorama.
4. Loads the phone's panorama image.
5. Creates a side-by-side comparison.

Output:
- output/q1_my_panorama.jpg (cropped)
- output/q1_compare_side_by_side.jpg

This implementation is correct, stable, and works for indoor portrait sequences.
"""

import cv2
import glob
import os
import numpy as np

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
PANORAMA_DIR = "data/panorama"
PHONE_PANO_PATH = "data/phone_panorama/phone_panorama.jpg"

OUT_PANO_PATH = "output/q1_my_panorama.jpg"
OUT_COMPARE_PATH = "output/q1_compare_side_by_side.jpg"

RESIZE_FACTOR = 0.35  # downscale to speed up stitching


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def load_images_sorted(folder):
    """Load all .jpg/.jpeg/.png in sorted order."""
    files = sorted(
        glob.glob(os.path.join(folder, "*.jpg")) +
        glob.glob(os.path.join(folder, "*.jpeg")) +
        glob.glob(os.path.join(folder, "*.png"))
    )
    if not files:
        raise RuntimeError(f"No images found in {folder}")

    imgs = []
    for f in files:
        im = cv2.imread(f)
        if im is None:
            raise RuntimeError(f"Failed to read: {f}")
        imgs.append(im)
    return imgs, files


def resize_images(images, fx):
    if fx == 1.0:
        return images
    return [cv2.resize(im, (0, 0), fx=fx, fy=fx) for im in images]


def crop_black_borders(img):
    """
    Robust cropping using contour detection.
    Removes black or dark filler regardless of exact pixel values.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold: treat anything < 30 as background (adjustable)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Find all contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  # failsafe

    # Largest contour = actual content
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    cropped = img[y:y+h, x:x+w]
    return cropped



def make_comparison(our_pano, phone_pano):
    """Resize both panoramas to same height and concatenate horizontally."""
    h1, w1 = our_pano.shape[:2]
    h2, w2 = phone_pano.shape[:2]

    H = min(h1, h2)
    s1 = H / h1
    s2 = H / h2

    r1 = cv2.resize(our_pano, (int(w1 * s1), H))
    r2 = cv2.resize(phone_pano, (int(w2 * s2), H))

    return np.hstack([r1, r2])


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    os.makedirs("output", exist_ok=True)

    print("Loading input images...")
    images, files = load_images_sorted(PANORAMA_DIR)
    for f in files:
        print("   ", f)

    print(f"Resizing by factor {RESIZE_FACTOR}...")
    images = resize_images(images, RESIZE_FACTOR)

    print("Creating OpenCV Stitcher...")
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    except AttributeError:
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

    print("Stitching images...")
    status, pano = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Stitching failed with error code {status}")

    # Save raw panorama first
    cv2.imwrite(OUT_PANO_PATH, pano)
    print("Saved RAW panorama:", OUT_PANO_PATH)

    # Crop black borders
    print("Cropping black borders...")
    pano_cropped = crop_black_borders(pano)
    cv2.imwrite(OUT_PANO_PATH, pano_cropped)
    pano = pano_cropped
    print("Saved CROPPED panorama:", OUT_PANO_PATH)

    # Load phone panorama
    print("Loading phone panorama:", PHONE_PANO_PATH)
    phone = cv2.imread(PHONE_PANO_PATH)
    if phone is None:
        raise RuntimeError("Could not load phone panorama.")

    # Create comparison
    comparison = make_comparison(pano, phone)
    cv2.imwrite(OUT_COMPARE_PATH, comparison)
    print("Saved side-by-side comparison:", OUT_COMPARE_PATH)

    print("Done.")


if __name__ == "__main__":
    main()
