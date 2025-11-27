import cv2
import glob
from src.stitcher import PanoramaStitcher

print("Loading images...")
files = sorted(glob.glob("data/panorama/*.jpg"))
print("Found:", files)

images = [cv2.imread(f) for f in files]
print("Loaded", len(images), "images")

print("Resizing images for speed...")
images = [cv2.resize(im, (0,0), fx=0.40, fy=0.40) for im in images]

stitcher = PanoramaStitcher()

print("Starting stitching...")
pano = stitcher.stitch(images)

print("Saving output...")
cv2.imwrite("output/panorama_opencv_baseline.jpg", pano)
print("DONE.")
