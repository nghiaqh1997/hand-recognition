import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import time
import math
import pyautogui
from sklearn.metrics.pairwise import euclidean_distances
def nothing(x):
    pass
cv2.namedWindow("Trackbars")
cv2.createTrackbar("bgSubThreshold", "Trackbars", 60, 200, nothing)
#cv2.createTrackbar("H_min", "Trackbars", 0, 179, nothing)
#cv2.createTrackbar("S_min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V_min", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("H_max", "Trackbars", 179, 179, nothing)
#cv2.createTrackbar("S_max", "Trackbars", 255, 255, nothing)
#cv2.createTrackbar("V_max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("gamma","Trackbars",100,300,nothing)
def remove_background(crop_image):
    fgmask = bgModel.apply(crop_image, learningRate=0)
    kernel = np.ones((5, 5), np.uint8)
    #fgmask = cv2.erode(fgmask, kernel, iterations=1)
    #fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)    
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel,iterations=3)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    res = cv2.bitwise_and(crop_image, crop_image, mask=fgmask)
    return res
cap = cv2.VideoCapture(0)
isBgCaptured = 0
game = 0
x=0
y=0
while True:
    gamma_change = cv2.getTrackbarPos("gamma","Trackbars")/100
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma_change) * 255.0, 0, 255)
    bgSubThreshold = cv2.getTrackbarPos("bgSubThreshold", "Trackbars")
    #H_min = cv2.getTrackbarPos("H_min", "Trackbars")
    H_min = 0
    #S_min = cv2.getTrackbarPos("S_min", "Trackbars")
    S_min = 0
    V_min = cv2.getTrackbarPos("V_min", "Trackbars")
    H_max = cv2.getTrackbarPos("H_max", "Trackbars")
    #S_max = cv2.getTrackbarPos("S_max", "Trackbars")
    S_max = 255
    #V_max = cv2.getTrackbarPos("V_max", "Trackbars")
    V_max = 255
    lower_red = np.array([H_min, S_min, V_min])
    upper_red = np.array([H_max, S_max, V_max])
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame,width=640)
    frame = cv2.LUT(frame, lookUpTable)
    frame = cv2.GaussianBlur(frame,(5,5),0)
    cv2.rectangle(frame, (320, 0), (640,240), (255,255,255), 2)
    crop_image = frame[0:240,320:640]
    if isBgCaptured == 1:
        img =  remove_background(crop_image)
        cv2.imshow('hehe',img)
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        #cv2.imshow('hehe1',hsv)
        YCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
        min_YCrCb = np.array([54, 131, 110])
        max_YCrCb = np.array([163, 157, 135])
        skinRegion = cv2.inRange(YCrCb,min_YCrCb,max_YCrCb)
        cv2.imshow('skinRegion',skinRegion)
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        #cv2.imshow("mask2", mask2)
        ret, thresh = cv2.threshold(mask2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU )
        #cv2.imshow("thresh", thresh)
        kernel = np.ones((5, 5), np.uint8)
        erotion = cv2.erode(thresh, kernel, iterations=2)
        dilation = cv2.dilate(erotion,kernel,iterations = 2)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel, iterations=3)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("closing", closing)
        contours = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contours = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
    try:
        contour = max(contours, key=lambda x: cv2.contourArea(x))#contour
        drawing = np.zeros(crop_image.shape, np.uint8) 
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (255, 100, 255), 0)
        #approximation
        #contour = sorted ( contours, key = cv2.contourArea, reverse = True ) [ : 1000 ]
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.001*peri , True)
        screenCnt = approx   
        #text = "original, num_pts={}".format(len(approx))
        #cv2.putText(drawing, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 255, 0), 2)
        #cv2.drawContours(drawing, [screenCnt], -1, (255, 255, 255), 5)
        cv2.drawContours(crop_image, [screenCnt], -1, (255, 255, 255), 5)
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        #cv2.circle(drawing, (cx, cy), 7, (255, 255, 255), -1)
        cv2.circle(crop_image, (cx, cy), 7, (255, 255, 255), -1)
        hull = cv2.convexHull(approx)
        # find the most extreme points in the convex hull
        extreme_top    = tuple(hull[hull[:, :, 1].argmin()][0])
        extreme_bottom = tuple(hull[hull[:, :, 1].argmax()][0])
        extreme_left   = tuple(hull[hull[:, :, 0].argmin()][0])
        extreme_right  = tuple(hull[hull[:, :, 0].argmax()][0])
        cv2.drawContours(drawing, [hull], -1, (0, 255, 255), 1)#
        M = [[cx,cy]]
        hull = hull.reshape(hull.shape[0],hull.shape[2])
        i = []
        for c in range(hull.shape[0]):
            a = hull[c,1]
            i = np.append(i,a)
        min_hull = min(i)
        a = np.where(i==min_hull)
        min_hull_index = a[0][0]
        min_x,min_y = hull[min_hull_index]
        MAX = max(euclidean_distances(hull,M))
        cv2.circle(crop_image, (cx,cy), int(MAX*0.6), [0, 0, 255], 2)
        #cv2.circle(drawing, (cx,cy), int(MAX*0.4), [0, 0, 255], 2)
        #cv2.imshow("Gestur", drawing)
        
        # find the center of the palm
        cX = int((extreme_left[0] + extreme_right[0]) / 2)
        cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
        #print("Center point : " + str(tuple((cX,cY))))
        #cv2.drawContours(image, [hull], -1, (0, 255, 0), 2)
        cv2.circle(crop_image, (cX, cY), radius=5, color=(255,0,0), thickness=5)
        #cv2.circle(crop_image, extreme_top, radius=5, color=(0,0,255), thickness=5)
        #cv2.circle(crop_image, extreme_bottom, radius=5, color=(0,0,255), thickness=5)
        #cv2.circle(crop_image, extreme_left, radius=5, color=(0,0,255), thickness=5)
        #cv2.circle(crop_image, extreme_right, radius=5, color=(0,0,255), thickness=5)
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)        
        count_defects = 0
        #print(min_x)
        #print(min_y)
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14 
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            d = 2*ar/a
            tan = math.atan2((cy-min_y),(cy-min_x))*180/3.14
            e = math.sqrt((min_x - cx) ** 2 + (min_y - cy) ** 2)
            if angle <= 90 and d > 45  :                
                count_defects += 1
                cv2.circle(crop_image, far, 2, [0, 0, 255], -1)
            print(e)    
            if count_defects == 0 and e < 95  :
                zero = 0
            if count_defects == 0 and e > 95  :
                zero = 1    
            cv2.line(crop_image, start, end, [255, 25, 100], 2) 
        if game == 1:
            if count_defects == 1:
                cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255), 5)
                """
                try:
                    pyautogui.press('up')
                except:
                    pass
                """
            elif count_defects == 2:
                cv2.putText(frame, "THREE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255), 5)
                """
                try:
                    pyautogui.press('down')
                except:
                    pass
                """
            elif count_defects == 3:
                cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,2 ,(255,0,255), 5)
                """
                try:
                    x, y = pyautogui.locateCenterOnScreen('exit.jpg')
                    pyautogui.moveTo(x, y, 2)
                    pyautogui.click(x, y)
                except:
                    pass
                """
            elif count_defects == 4:
                cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255), 5)
                """
                try:
                    x, y = pyautogui.locateCenterOnScreen('retry.jpg')
                    pyautogui.moveTo(x, y, 2)
                    pyautogui.click(x, y)
                except:
                    pass
                """
            elif count_defects == 0 and zero!= 0:
                cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255),5)
                """
                try:
                    x, y = pyautogui.locateCenterOnScreen('level.jpg')
                    if x == 0:
                        x, y = pyautogui.locateCenterOnScreen('car.jpg')
                    if x == 0:
                        x, y = pyautogui.locateCenterOnScreen('enter.jpg')
                    pyautogui.moveTo(x, y, 2)
                    pyautogui.click(x, y)
                except:
                    pass
                """
            elif zero == 0:
                cv2.putText(frame, "ZERO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255),5)
            else:
                pass
        else:
            pass 
                
    except:
        pass
    cv2.imshow('img1',frame)
    key_press = cv2.waitKey(100)
    if key_press == ord('q'):
        break
    elif key_press == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(30, bgSubThreshold,detectShadows=False)
        isBgCaptured = 1
        time.sleep(2)
    elif key_press == ord('r'):
        bgModel = None
        isBgCaptured = 0
        time.sleep(1)
    elif key_press == ord('z'):
        game = 1
    
cap.release()
cv2.destroyAllWindows
