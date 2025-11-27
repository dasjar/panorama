#!/usr/bin/env python3
"""
q2_sift_ransac_demo.py

CSC 8830 – CV
Assignment 4 – Q2

- Run SIFT-from-scratch on a pair of images.
- Match descriptors via BFMatcher.
- Estimate homography via RANSAC.
- Warp second image onto first.
- Compare with OpenCV SIFT (if available).
"""

import os
import cv2
import numpy as np

from src.sift_from_scratch import SIFTFromScratch

PAIR_DIR = "data/sift_pair"
IMG1_PATH = os.path.join(PAIR_DIR, "1.jpg")
IMG2_PATH = os.path.join(PAIR_DIR, "2.jpg")


OUT_MATCHES_MY = "output/q2_my_sift_matches.jpg"
OUT_WARPED_MY = "output/q2_my_sift_warped.jpg"

OUT_MATCHES_CV = "output/q2_cv_sift_matches.jpg"
OUT_WARPED_CV = "output/q2_cv_sift_warped.jpg"


def match_descriptors(desc1, desc2, k=2, ratio=0.75, use_l2=True):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    norm = cv2.NORM_L2 if use_l2 else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm)
    raw_matches = bf.knnMatch(desc1, desc2, k=k)

    good = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            good.append(m[0])
    return good


def ransac_homography(kps1_xy, kps2_xy, matches, reproj_thresh=4.0):
    """
    Compute H (img1 -> img2) using matched points and RANSAC.
    kps1_xy, kps2_xy: np.array [N, 2] of xy coordinates
    matches: list of cv2.DMatch (queryIdx: idx in desc1, trainIdx: idx in desc2)
    """
    if len(matches) < 4:
        return None, None

    pts1 = np.float32([kps1_xy[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2_xy[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)
    return H, mask


def keypoints_to_xy(my_kpts):
    """
    Convert our keypoint dicts into [N, 2] (x,y) array.
    """
    coords = []
    for kp in my_kpts:
        coords.append([kp["x"], kp["y"]])
    return np.array(coords, dtype=np.float32)


def draw_matches(img1, img2, pts1, pts2, matches, mask):
    """
    Draw inlier matches on a side-by-side canvas.
    pts1, pts2: arrays [N, 2]
    matches: list of cv2.DMatch
    mask: inlier mask from RANSAC
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[0:h1, 0:w1] = img1
    canvas[0:h2, w1:w1 + w2] = img2

    for m, inlier in zip(matches, mask.ravel()):
        if not inlier:
            continue
        x1, y1 = pts1[m.queryIdx]
        x2, y2 = pts2[m.trainIdx]
        x2_shift = x2 + w1

        color = (0, 255, 0)
        cv2.circle(canvas, (int(x1), int(y1)), 3, color, -1)
        cv2.circle(canvas, (int(x2_shift), int(y2)), 3, color, -1)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2_shift), int(y2)), color, 1)

    return canvas


def run_my_sift(img1, img2):
    print("[INFO] Running SIFT-from-scratch...")
    sift = SIFTFromScratch()

    kps1, desc1 = sift.detectAndCompute(img1)
    kps2, desc2 = sift.detectAndCompute(img2)

    print(f"[INFO] My SIFT: {len(kps1)} keypoints in img1, {len(kps2)} in img2")

    pts1 = keypoints_to_xy(kps1)
    pts2 = keypoints_to_xy(kps2)

    matches = match_descriptors(desc1, desc2, ratio=0.75, use_l2=True)
    print(f"[INFO] My SIFT: {len(matches)} matches after ratio test")

    H, mask = ransac_homography(pts1, pts2, matches, reproj_thresh=4.0)
    if H is None:
        print("[WARN] My SIFT: RANSAC could not find a homography.")
        return

    print("[INFO] My SIFT: RANSAC inliers:", int(mask.sum()))

    # Draw inliers
    match_vis = draw_matches(img1, img2, pts1, pts2, matches, mask)
    cv2.imwrite(OUT_MATCHES_MY, match_vis)
    print("[INFO] Saved:", OUT_MATCHES_MY)

    # Warp img1 into img2 frame
    h2, w2 = img2.shape[:2]
    warp = cv2.warpPerspective(img1, H, (w2, h2))
    blended = 0.5 * warp + 0.5 * img2
    blended = blended.astype(np.uint8)
    cv2.imwrite(OUT_WARPED_MY, blended)
    print("[INFO] Saved:", OUT_WARPED_MY)


def run_cv_sift(img1, img2):
    if not hasattr(cv2, "SIFT_create"):
        print("[WARN] OpenCV SIFT is not available in this build. Skipping CV-SIFT comparison.")
        return

    print("[INFO] Running OpenCV SIFT...")
    sift = cv2.SIFT_create()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kps1, desc1 = sift.detectAndCompute(gray1, None)
    kps2, desc2 = sift.detectAndCompute(gray2, None)

    print(f"[INFO] CV SIFT: {len(kps1)} keypoints in img1, {len(kps2)} in img2")

    pts1 = np.float32([kp.pt for kp in kps1])
    pts2 = np.float32([kp.pt for kp in kps2])

    matches = match_descriptors(desc1, desc2, ratio=0.75, use_l2=True)
    print(f"[INFO] CV SIFT: {len(matches)} matches after ratio test")

    H, mask = ransac_homography(pts1, pts2, matches, reproj_thresh=4.0)
    if H is None:
        print("[WARN] CV SIFT: RANSAC could not find a homography.")
        return

    print("[INFO] CV SIFT: RANSAC inliers:", int(mask.sum()))

    match_vis = draw_matches(img1, img2, pts1, pts2, matches, mask)
    cv2.imwrite(OUT_MATCHES_CV, match_vis)
    print("[INFO] Saved:", OUT_MATCHES_CV)

    h2, w2 = img2.shape[:2]
    warp = cv2.warpPerspective(img1, H, (w2, h2))
    blended = 0.5 * warp + 0.5 * img2
    blended = blended.astype(np.uint8)
    cv2.imwrite(OUT_WARPED_CV, blended)
    print("[INFO] Saved:", OUT_WARPED_CV)


def main():
    os.makedirs("output", exist_ok=True)

    img1 = cv2.imread(IMG1_PATH)
    img2 = cv2.imread(IMG2_PATH)
    if img1 is None or img2 is None:
        raise RuntimeError("Could not read img1/img2 in data/sift_pair/")

    run_my_sift(img1, img2)
    run_cv_sift(img1, img2)

    print("[INFO] Q2 SIFT + RANSAC demo finished.")


if __name__ == "__main__":
    main()
