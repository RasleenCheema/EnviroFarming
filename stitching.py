import cv2
import numpy as np

img1 = cv2.imread('C:/Users/Dell/PycharmProjects/Sabudh/8.JPG')
print('Original Dimensions : ', img1.shape)
assert not isinstance(img1,type(None)), 'image not found'


scale_percent = 0.05 # percent of original size
width = int(img1.shape[1] * scale_percent )
height = int(img1.shape[0] * scale_percent)
dim = (width, height)
# resize image
resized1= cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
print('Resized Dimensions1 : ', resized1.shape)
cv2.imshow("Resized image1", resized1)

img2 = cv2.imread("C:/Users/Dell/PycharmProjects/Sabudh/7.JPG")

assert not isinstance(img2,type(None)), 'image not found'
print('Original Dimensions 2: ',img2.shape)

scale_percent = 0.05 # percent of original size
width = int(img2.shape[1] * scale_percent )
height = int(img2.shape[0] * scale_percent)
dim = (width, height)
# resize image
resized2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
print('Resized Dimensions2 : ', resized2.shape)
cv2.imshow("Resized image2", resized2)

img3 = cv2.imread('C:/Users/Dell/PycharmProjects/Sabudh/DJI_0695.JPG')
print('Original Dimensions : ', img3.shape)
assert not isinstance(img3,type(None)), 'image not found'


scale_percent = 0.05# percent of original size
width = int(img3.shape[1] * scale_percent )
height = int(img3.shape[0] * scale_percent)
dim = (width, height)
# resize image
resized3= cv2.resize(img3, dim, interpolation=cv2.INTER_AREA)
print('Resized Dimensions3: ', resized3.shape)
cv2.imshow("Resized image3", resized3)

img4 = cv2.imread('C:/Users/Dell/PycharmProjects/Sabudh/DJI_0684.JPG')
print('Original Dimensions : ', img4.shape)
assert not isinstance(img4,type(None)), 'image not found'


scale_percent = 0.05# percent of original size
width = int(img4.shape[1] * scale_percent )
height = int(img4.shape[0] * scale_percent)
dim = (width, height)
# resize image
resized4= cv2.resize(img4, dim, interpolation=cv2.INTER_AREA)
print('Resized Dimensions4 : ', resized4.shape)
cv2.imshow("Resized image4", resized4)




# resize image
# img2 = imutils.resize(img2, width=1280)
# cv2.imshow('image' , img2)
# img1.resize(2000,1000)
# img2.resize(2000,1000)
# ORB Detectorme_holding_book.jpg
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
kp3, des3 = orb.detectAndCompute(img3, None)
kp4, des4 = orb.detectAndCompute(img4, None)

for d in des1:
    print(d)
# Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key = lambda x:x.distance)
matching_result = cv2.drawMatches(resized1, kp1, resized2, kp2, matches[:0], None, flags=2)
cv2.imshow("Matching result", matching_result)

bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches1 = bf1.match(des3, des4)

matches1 = sorted(matches1, key = lambda x:x.distance)
matching_result1 = cv2.drawMatches(resized3, kp3, resized4, kp4, matches1[:0], None, flags=2)
kp5,des5=orb.detectAndCompute(matching_result,None)
kp6,des6=orb.detectAndCompute(matching_result1,None)


cv2.imshow("Matching result1", matching_result1)
bf2 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches2 = bf2.match(des5, des6)
matches2 = sorted(matches2, key = lambda x:x.distance)
matching_result2 = cv2.drawMatches(matching_result, kp5, matching_result1, kp6, matches2[:0], None, flags=2)

cv2.imshow("Matching result2", matching_result2)

cv2.waitKey(0)
cv2.destroyAllWindows()
