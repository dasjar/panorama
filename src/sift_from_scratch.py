#!/usr/bin/env python3
"""
sift_from_scratch.py

Simplified SIFT implementation from scratch:
- Gaussian pyramid
- DoG pyramid
- Local extrema in scale space
- Contrast threshold
- Orientation assignment
- 128-d descriptor (4x4 cells, 8 bins)

Returns: keypoints (y, x, scale, orientation) and descriptors.
"""

import numpy as np
import cv2


class SIFTFromScratch:
    def __init__(
        self,
        num_octaves: int = 4,
        scales_per_octave: int = 3,
        sigma0: float = 1.6,
        contrast_thresh: float = 0.03,
        edge_thresh: float = 10.0
    ):
        self.num_octaves = num_octaves
        self.scales_per_octave = scales_per_octave
        self.sigma0 = sigma0
        self.contrast_thresh = contrast_thresh
        self.edge_thresh = edge_thresh

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detectAndCompute(self, image_gray: np.ndarray):
        """
        Main entry point.
        image_gray: single-channel float32 or uint8 [H, W].

        Returns:
            keypoints: list of dicts with (y, x, octave, scale, sigma, angle)
            descriptors: np.ndarray [N, 128]
        """
        if image_gray.ndim == 3:
            image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
        if image_gray.dtype != np.float32:
            image_gray = image_gray.astype(np.float32) / 255.0

        gauss_pyr, sigmas = self._build_gaussian_pyramid(image_gray)
        dog_pyr = self._build_dog_pyramid(gauss_pyr)

        raw_kpts = self._detect_keypoints(dog_pyr, sigmas)

        kpts_oriented = self._assign_orientations(gauss_pyr, raw_kpts)
        descs = self._compute_descriptors(gauss_pyr, kpts_oriented)

        return kpts_oriented, descs

    # ------------------------------------------------------------------
    # Pyramid construction
    # ------------------------------------------------------------------
    def _build_gaussian_pyramid(self, base_img):
        """
        Build Gaussian pyramid.

        gauss_pyr[octave][scale] = Gaussian-blurred image
        sigmas[octave][scale] = sigma for that image
        """
        k = 2 ** (1.0 / self.scales_per_octave)
        num_levels = self.scales_per_octave + 3  # extra for DoG

        gauss_pyr = []
        sigmas_pyr = []

        # Initial image: doubled or original; keep simple: original
        base = base_img.copy()

        for o in range(self.num_octaves):
            octave_imgs = []
            octave_sigmas = []

            sigma_prev = 0.5  # nominal sigma of input
            for s in range(num_levels):
                sigma_total = self.sigma0 * (k ** s)
                sigma = np.sqrt(max(sigma_total**2 - sigma_prev**2, 1e-10))
                blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma, sigmaY=sigma)
                octave_imgs.append(blurred)
                octave_sigmas.append(sigma_total)
                sigma_prev = sigma_total

            gauss_pyr.append(octave_imgs)
            sigmas_pyr.append(octave_sigmas)

            # Next octave base = downsample by 2
            h, w = base.shape
            base = cv2.resize(base, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)

        return gauss_pyr, sigmas_pyr

    def _build_dog_pyramid(self, gauss_pyr):
        """
        Build Difference-of-Gaussians pyramid.
        dog_pyr[oct][s] = G(o,s+1) - G(o,s)
        """
        dog_pyr = []
        for octave_imgs in gauss_pyr:
            dogs = []
            for i in range(1, len(octave_imgs)):
                dogs.append(octave_imgs[i] - octave_imgs[i - 1])
            dog_pyr.append(dogs)
        return dog_pyr

    # ------------------------------------------------------------------
    # Keypoint detection
    # ------------------------------------------------------------------
    def _detect_keypoints(self, dog_pyr, sigmas_pyr):
        """
        Find local extrema in 3x3x3 neighborhoods in DoG pyramids,
        apply contrast threshold, and return list of raw keypoints.
        """
        keypoints = []

        for o, dogs in enumerate(dog_pyr):
            # dogs: [num_levels-1] images
            for s in range(1, len(dogs) - 1):
                dog_prev = dogs[s - 1]
                dog = dogs[s]
                dog_next = dogs[s + 1]

                h, w = dog.shape
                # avoid borders
                for y in range(1, h - 1):
                    for x in range(1, w - 1):
                        val = dog[y, x]
                        if abs(val) < self.contrast_thresh:
                            continue

                        patch_prev = dog_prev[y - 1:y + 2, x - 1:x + 2]
                        patch = dog[y - 1:y + 2, x - 1:x + 2]
                        patch_next = dog_next[y - 1:y + 2, x - 1:x + 2]
                        neighbors = np.hstack([
                            patch_prev.flatten(),
                            patch.flatten(),
                            patch_next.flatten()
                        ])

                        center = neighbors[13]  # middle of 27
                        neighbors = np.delete(neighbors, 13)

                        if (val > 0 and val >= neighbors.max()) or (
                            val < 0 and val <= neighbors.min()
                        ):
                            # extremum
                            sigma = sigmas_pyr[o][s + 1]  # correspond to G(o,s+1)
                            keypoints.append({
                                "octave": o,
                                "scale": s,
                                "y": y,
                                "x": x,
                                "sigma": sigma,
                                "angle": 0.0,  # to be assigned later
                            })

        return keypoints

    # ------------------------------------------------------------------
    # Orientation assignment
    # ------------------------------------------------------------------
    def _assign_orientations(self, gauss_pyr, keypoints):
        """
        For each keypoint in the Gaussian image at its octave/scale,
        compute gradient magnitude and orientation in a window, build
        orientation histogram, and assign dominant angle.
        """
        oriented = []
        num_bins = 36
        for kp in keypoints:
            o = kp["octave"]
            s = kp["scale"] + 1  # DoG index shift -> Gaussian index
            img = gauss_pyr[o][s]
            h, w = img.shape
            y = kp["y"]
            x = kp["x"]

            if x <= 1 or x >= w - 2 or y <= 1 or y >= h - 2:
                continue

            # Local gradient
            # Use a region of radius ~3 * sigma
            sigma = kp["sigma"]
            radius = int(round(3 * sigma))
            y0 = max(y - radius, 1)
            y1 = min(y + radius, h - 2)
            x0 = max(x - radius, 1)
            x1 = min(x + radius, w - 2)

            patch = img[y0:y1 + 1, x0:x1 + 1]
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)

            mag = np.sqrt(gx**2 + gy**2)
            ang = (np.rad2deg(np.arctan2(gy, gx)) + 360.0) % 360.0

            # Orientation histogram
            hist = np.zeros(num_bins, dtype=np.float32)
            bin_width = 360.0 / num_bins

            # weight by Gaussian in spatial domain
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            yy = yy - y
            xx = xx - x
            w_spatial = np.exp(-(xx**2 + yy**2) / (2 * (1.5 * sigma) ** 2))

            weighted_mag = mag * w_spatial

            for yy_i in range(weighted_mag.shape[0]):
                for xx_i in range(weighted_mag.shape[1]):
                    m = weighted_mag[yy_i, xx_i]
                    a = ang[yy_i, xx_i]
                    bin_idx = int(np.floor(a / bin_width)) % num_bins
                    hist[bin_idx] += m

            # dominant orientation
            max_bin = np.argmax(hist)
            angle = max_bin * bin_width

            new_kp = kp.copy()
            new_kp["angle"] = angle
            oriented.append(new_kp)

        return oriented

    # ------------------------------------------------------------------
    # Descriptor computation
    # ------------------------------------------------------------------
    def _compute_descriptors(self, gauss_pyr, keypoints):
        """
        4x4 cells, 8 orientation bins => 128-d descriptor per keypoint.
        """
        descriptors = []
        for kp in keypoints:
            o = kp["octave"]
            s = kp["scale"] + 1
            img = gauss_pyr[o][s]
            h, w = img.shape
            y = kp["y"]
            x = kp["x"]
            angle = kp["angle"]
            sigma = kp["sigma"]

            # Descriptor window ~ 4 x 4 cells, each 4x4 pixels => 16x16
            win_size = int(round(16 * sigma / self.sigma0))
            if win_size < 16:
                win_size = 16

            half = win_size // 2
            y0 = max(y - half, 1)
            y1 = min(y + half, h - 2)
            x0 = max(x - half, 1)
            x1 = min(x + half, w - 2)

            patch = img[y0:y1 + 1, x0:x1 + 1]
            if patch.shape[0] < 16 or patch.shape[1] < 16:
                continue

            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx**2 + gy**2)
            ang = (np.rad2deg(np.arctan2(gy, gx)) + 360.0) % 360.0

            # rotate angles relative to keypoint orientation
            ang = (ang - angle + 360.0) % 360.0

            # 4x4 cells, 8 bins
            desc = np.zeros((4, 4, 8), dtype=np.float32)
            cell_h = patch.shape[0] / 4.0
            cell_w = patch.shape[1] / 4.0
            bin_width = 360.0 / 8

            for yy in range(patch.shape[0]):
                for xx in range(patch.shape[1]):
                    m = mag[yy, xx]
                    a = ang[yy, xx]
                    cy = int(yy / cell_h)
                    cx = int(xx / cell_w)
                    if cy >= 4 or cx >= 4:
                        continue
                    b = int(a / bin_width) % 8
                    desc[cy, cx, b] += m

            vec = desc.flatten()

            # Normalize and threshold
            norm = np.linalg.norm(vec) + 1e-10
            vec = vec / norm
            vec = np.clip(vec, 0, 0.2)
            norm = np.linalg.norm(vec) + 1e-10
            vec = vec / norm

            descriptors.append(vec)

        if len(descriptors) == 0:
            return np.zeros((0, 128), dtype=np.float32)

        return np.vstack(descriptors).astype(np.float32)
