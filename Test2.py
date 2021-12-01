import numpy as np
import cv2
import math
from datetime import datetime, timedelta

def colorMask(frame): #colorMask is created for highlighting specific colors in the image
    #Greenscreen grøn HSV/HSB values = 120°, 98%, 96%
    #Greenscreen grøn RGB values = 4, 244, 4

    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Change default BGR colors to HSV
    lowerThresh = np.array([0, 48, 80], dtype = "uint8") #Lower range of color,
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
kernel = np.ones((8,8),np.uint8)

period = timedelta(seconds = 1)
next_time = datetime.now() + period
seconds = 0
count = 0

while video.isOpened():
    _, frame = video.read()
    #frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (50, 100), (250, 350), (255, 0, 0), 0)
    crop_image = frame[100:350, 50:250]

    try:
        mask_img = colorMask(frame)
        imageD = cv2.dilate(mask_img, kernel, iterations=1)
        imageE = cv2.erode(imageD, kernel, iterations=1)
        contours, hull = getContour(imageE)
        cv2.drawContours(frame, [contours], -1, (255,255,0), 2) #Make a cyan contour
        cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2) #Make a yellow convex hull
        defects = getDefects(contours)

        #Her skal der være noget cosinus relation matematik magic for at detect specific gesture
        #Brug defects og contour + noget andet cosinus BS YEP
        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(contours[s][0])
                end = tuple(contours[e][0])
                far = tuple(contours[f][0])
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #      cosine theorem
                if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(frame, far, 4, [0, 0, 255], -1)
            if cnt > 0:
                cnt = cnt+1
            cv2.putText(frame, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)

            while next_time < datetime.now():
                seconds += 1
                next_time += period
                print(seconds)

                if count != cnt:
                    count = cnt
                    seconds = 0

                if count == 0:
                    if seconds == 2:
                        print("0 fingre")
                        seconds = 0

                if count == 1:
                    if seconds == 2:
                        print("1 finger")
                        seconds = 0

                if count == 2:
                    if seconds == 2:
                        print("2 fingre")
                        seconds = 0

                if count == 3:
                    if seconds == 2:
                        print("3 fingre")
                        seconds = 0

                if count == 4:
                    if seconds == 2:
                        print("4 fingre")
                        seconds = 0

                if count == 5:
                    if seconds == 2:
                        print("5 fingre")
                        seconds = 0

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask_img)
        cv2.imshow("frame2", imageE)
    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()