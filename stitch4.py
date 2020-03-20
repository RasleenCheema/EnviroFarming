import glob
import cv2 as cv
import numpy as np

path = glob.glob("C:/Users/Dell/PycharmProjects/Sabudh/New folder (7)/*.JPG")
print("hii")
cv_img = []
images = []
for img in path:
    dim = (1024, 768)
    n = cv.imread(img)
    assert not isinstance(img, type(None)), 'image not found'
    scale_percent = 0.05  # percent of original size
    width = int(n.shape[1] * scale_percent)
    height = int(n.shape[0] * scale_percent)
    dim = (width, height)
    img1 = cv.resize(n, dim, interpolation=cv.INTER_AREA)
    cv_img.append(n)
    print(img)
    images.append(img1)
stitcher = cv.Stitcher.create()
ret, pano = stitcher.stitch(images)


if ret == cv.STITCHER_OK:
    cv.imshow('Stitched_image', pano)
    cv.waitKey(500000)
    cv.destroyAllWindows()

