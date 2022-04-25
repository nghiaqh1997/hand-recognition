# Imports
import numpy as np
import cv2
import math
def nothing(x):
    pass
# Open Camera
capture = cv2.VideoCapture(1)
cv2.namedWindow("Trackbars")
cv2.createTrackbar("H_min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("S_min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V_min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("H_max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S_max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V_max", "Trackbars", 255, 255, nothing)
while capture.isOpened():
    H_min = cv2.getTrackbarPos("H_min", "Trackbars")
    S_min = cv2.getTrackbarPos("S_min", "Trackbars")
    V_min = cv2.getTrackbarPos("V_min", "Trackbars")
    H_max = cv2.getTrackbarPos("H_max", "Trackbars")
    S_max = cv2.getTrackbarPos("S_max", "Trackbars")
    V_max = cv2.getTrackbarPos("V_max", "Trackbars")
    lower_red = np.array([H_min, S_min, V_min])
    upper_red = np.array([H_max, S_max, V_max])

    
    ret, frame = capture.read()

    frame = cv2.flip(frame,1)

    cv2.rectangle(frame, (0, 0), (300, 300), (0, 255, 0), 0)
    crop_image = frame[0:300, 0:300]
    #cv2.imshow("crop_image", crop_image)

   
    blur = cv2.GaussianBlur(crop_image, (9, 9), 0)
    #cv2.imshow("blur", blur)

    
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv", hsv)

    
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    #mask2 = cv2.inRange(hsv, np.array([0, 21, 95]), np.array([33, 255, 255]))
    #cv2.imshow("mask2", mask2)

    
    kernel = np.ones((5, 5))

    
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    
    filtered = cv2.GaussianBlur(erosion, (9, 9), 0)
    ret, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    
    cv2.imshow("Thresholded Hemal", thresh)

    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    try:
       
        
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (255, 100, 255), 0)

        
        hull = cv2.convexHull(contour)

        
        drawing = np.zeros(crop_image.shape, np.uint8)
        
        cv2.drawContours(drawing, [contour], -1, (255, 255, 0), 0)#green
        cv2.drawContours(drawing, [hull], -1, (0, 255, 255), 0)#yellow
        cv2.imshow("Gestur", drawing)
       
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 25, 100], 2)

        
        if count_defects == 0:
            cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255),5)
        elif count_defects == 1:
            cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255), 5)
        elif count_defects == 2:
            cv2.putText(frame, "THREE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255), 5)
        elif count_defects == 3:
            cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,2 ,(255,0,255), 5)
        elif count_defects == 4:
            cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255), 5)
        else:
            pass
    except:
        pass

   
    cv2.imshow("Gesture", frame)
    

    
    if cv2.waitKey(10) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
