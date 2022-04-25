import cv2, pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3
import imutils
import pyautogui
def gui_py():
    button_reset_game = pyautogui.locateOnScreen('img_game/2.bmp')
    #print(button_reset_game)
    button_reset_game = pyautogui.center(button_reset_game)
    #print(button_reset_game)
    button_reset_game_x,button_reset_game_y = button_reset_game
    #print(button_reset_game_x)
    #print(button_reset_game_y)
    pyautogui.moveTo(button_reset_game_x,button_reset_game_y,5)
    pyautogui.click(button_reset_game_x,button_reset_game_y,5)
#gui_py()
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.autograph.set_verbosity(0)
prediction = None
model = tf.keras.models.load_model('cnn_model_keras2.h5')
def get_image_size():
    img = cv2.imread('gestures/0/100.jpg', 0)
    return img.shape
# 50,50
image_x, image_y = get_image_size()
def tf_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    np_array = np.array(img)
    return np_array
# shape 50,50
def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
    cursor = conn.execute(cmd)
    for row in cursor:
        return row[0]

def split_sentence(text, num_of_words):
    '''
    Splits a text into group of num_of_words
    '''
    list_words = text.split(" ")
    length = len(list_words)
    splitted_sentence = []
    b_index = 0
    e_index = num_of_words
    while length > 0:
        part = ""
        for word in list_words[b_index:e_index]:
            part = part + " " + word
        splitted_sentence.append(part)
        b_index += num_of_words
        e_index += num_of_words
        length -= num_of_words
    return splitted_sentence

def put_splitted_text_in_blackboard(blackboard, splitted_text):
    y = 200
    for text in splitted_text:
        cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        y += 50
def recognize():
    global prediction
    aWeight = 0.5
    top, right, bottom, left = 120, 0, 350, 200
    num_frames = 0
    cam = cv2.VideoCapture(0)
    while True:
        text = ""
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
        num_frames += 1
        cv2.imshow("Video", clone) 
        try:
            if len(contours) > 0:
                contour = max(contours, key = cv2.contourArea)
                #print(cv2.contourArea(contour))
                if cv2.contourArea(contour) > 10000:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    save_img = thresh[y1:y1+h1, x1:x1+w1]
                    
                    if w1 > h1:
                        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                    elif h1 > w1:
                        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
                    
                    pred_probab, pred_class = keras_predict(model, save_img)
                    
                    if pred_probab*100 > 80:
                        text = get_pred_text_from_db(pred_class)
                        print(text)
                    if text == '9':
                        print("hehe")
                    elif text =='10':
                        print("hoho")
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            splitted_text = split_sentence(text, 2)
            put_splitted_text_in_blackboard(blackboard, splitted_text)
            #cv2.putText(blackboard, text, (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (255, 255, 255))
            res = np.hstack((img, blackboard))
            cv2.imshow("Recognizing gesture", res)
            cv2.imshow("thresh", thresh)
        except:
            pass
        #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        
        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

bg = None
keras_predict(model, np.zeros((50, 50), dtype=np.uint8))		
recognize()
cam.release()
cv2.destroyAllWindows()