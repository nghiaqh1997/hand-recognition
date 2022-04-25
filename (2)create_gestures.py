import cv2
import numpy as np
import pickle, os, sqlite3, random

image_x, image_y = 50, 50
def init_create_folder_database():
    # create the folder and database if not exist
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()
def create_folder(folder_name):
	if not os.path.exists(folder_name):
		os.mkdir(folder_name)
def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (%s, \'%s\')" % (g_id, g_name)
    try:
        conn.execute(cmd)
    except sqlite3.IntegrityError:
        choice = input("g_id already exists. Want to change the record? (y/n): ")
        if choice.lower() == 'y':
            cmd = "UPDATE gesture SET g_name = \'%s\' WHERE g_id = %s" % (g_name, g_id)
            conn.execute(cmd)
        else:
            print("Doing nothing...")
            return
    conn.commit()
def run_avg(image,aWeight):
    global bg
    if bg is None :
        bg = image.copy().astype("float")
        return 
    cv2.accumulateWeighted(image,bg,aWeight)
def segment(image, threshold=30):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    ret, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    #ret, thresholded = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def store_images(g_id):
    total_pics = 1200
    cam = cv2.VideoCapture(0)
    #x, y, w, h = 20, 120, 270, 200
    create_folder("gestures/"+str(g_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0
    aWeight = 0.5
    top, right, bottom, left = 120, 0, 350, 200
    num_frames = 0
    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        clone = img.copy()
        roi = img[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
                thresh = thresholded
                # draw the segmented hand
                cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
                contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
                # increment the number of frames
        if frames>30:
            print(frames) 
        try:
            if len(contours) > 0:
                contour = max(contours, key = cv2.contourArea)
                if cv2.contourArea(contour) > 5000 and frames > 50:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    pic_no += 1
                    save_img = thresh[y1:y1+h1, x1:x1+w1]
                    if w1 > h1:
                        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , 
                                                      int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                    elif h1 > w1:
                        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , 
                                                      int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
                    save_img = cv2.resize(save_img, (image_x, image_y))
                    rand = random.randint(0, 10)
                    if rand % 2 == 0:
                        save_img = cv2.flip(save_img, 1)
                    cv2.putText(clone, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                    cv2.imwrite("gestures/"+str(g_id)+"/"+str(pic_no)+".jpg", save_img)
                    cv2.imshow("thresh", thresh)
            else:
                pass
        except:
            pass
        num_frames += 1
        #cv2.imshow("Video", clone)
        #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(clone, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", clone)
        
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            if flag_start_capturing == False:
                flag_start_capturing = True
            else:
                flag_start_capturing = False
                frames = 0
        if flag_start_capturing == True:
            frames += 1
        if pic_no == total_pics:
            break
        keypress_1 = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress_1 == ord("q"):
            break
bg = None
init_create_folder_database()
g_id = input("Enter gesture no.: ")
g_name = input("Enter gesture name/text: ")
store_in_db(g_id, g_name)
store_images(g_id)