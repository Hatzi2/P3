import numpy as np
import cv2

def colorMask(img):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerThresh = np.array([70, 70, 70], dtype = "uint8")
    upperThresh = np.array([120, 98, 96], dtype = "uint8")
    colorRegionHSV = cv2.inRange(hsvImg, lowerThresh, upperThresh)
    blurred = cv2.blur(colorRegionHSV, (2,2))

    #rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #lowerThresh = np.array([0, 0, 0], dtype = "uint8")
    #upperThresh = np.array([0, 255, 0], dtype = "uint8")
    #colorRegionRGB = cv2.inRange(rgbImg, lowerThresh, upperThresh)
    #blurred = cv2.blur(colorRegionRGB, (2,2))

    ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
    return thresh

def getContour(mask_img):
    contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key = lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(contours)
    return contours, hull

def getDefects(contours):
    hull = cv2.convexHull(contours, returnPoints = False)
    defects = cv2.convexityDefects(contours, hull)
    return defects

video = cv2.VideoCapture(2) # '0' for webcam
while video.isOpened():
    _, img = video.read()
    try:
        mask_img = colorMask(img)
        contours, hull = getContour(mask_img)
        cv2.drawContours(img, [contours], -1, (255,255,0), 2)
        cv2.drawContours(img, [hull], -1, (0, 255, 255), 2)
        defects = getDefects(contours)

        cv2.imshow("img", img)
    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
