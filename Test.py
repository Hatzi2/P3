import numpy as np
import cv2
import time

def colorMask(img): #colorMask is created for highlighting specific colors in the image
    #Greenscreen grøn HSV/HSB values = 120°, 98%, 96%
    #Greenscreen grøn RGB values = 4, 244, 4

    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Change default BGR colors to HSV
    lowerThresh = np.array([0, 48, 80], dtype = "uint8") #Lower range of color
    upperThresh = np.array([20, 255, 255], dtype = "uint8") #Upper range of color
    colorRegionHSV = cv2.inRange(hsvImg, lowerThresh, upperThresh) #Detect color on range of lower and upper pixel values in the colorspace
    blurred = cv2.blur(colorRegionHSV, (2,2)) #Blur image to improve masking

    ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY) #Apply threshing
    return thresh

def getContour(mask_img):
    contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key = lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(contours)
    return contours, hull

#Any defiation from the hull should be considered a convex defect
def getDefects(contours):
    hull = cv2.convexHull(contours, returnPoints = False)
    defects = cv2.convexityDefects(contours, hull)
    return defects

video = cv2.VideoCapture(0) # '0' for webcam
fgbg = cv2.createBackgroundSubtractorMOG2()

while video.isOpened():
    _, img = video.read()
    ret, frame = video.read()

    fgmask = fgbg.apply(frame)

    try:
        mask_img = colorMask(img)
        contours, hull = getContour(mask_img)
        cv2.drawContours(img, [contours], -1, (255,255,0), 2) #Make a cyan contour
        cv2.drawContours(img, [hull], -1, (0, 255, 255), 2) #Make a yellow convex hull
        defects = getDefects(contours)

        #Her skal der være noget cosinus relation matematik magic for at detect specific gesture
        #Brug defects og contour + noget andet cosinus BS YEP

    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("img", img)
    cv2.imshow("blur", mask_img)
    cv2.imshow("frame", fgmask)

video.release()
cv2.destroyAllWindows()
