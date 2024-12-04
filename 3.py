
# MotionScript project to convert Sign Language into coherent voice
# sometimes the autocorrect could make errors in converting wrong words to different words

# Implementation of LLM is remaining as of 13 April 2024.


# i have added 2 buttons , tab to get whitespaces and Q for delete option so it becomes easier
# for the deaf individual to perform whitspace and delete options smoothly

import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp



from gtts import gTTS
import os

def text_to_speech(text):

    tts = gTTS(text=text, lang='en')
    

    tts.save("output.mp3")
    

    if os.name == 'posix':
    
        os.system("afplay output.mp3")
    elif os.name == 'nt':
 
        os.system("start output.mp3")
    else:

        os.system("aplay output.mp3")

from utils import CvFpsCalc
from model import MotionscriptModel
from model import PointHistoryClassifier
from gtts import gTTS
import os
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_type_of_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

concatenated_hand_sign_text = ""
def main():
  
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_type_of_mode = args.use_static_image_type_of_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_type_of_mode=use_static_image_type_of_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = MotionscriptModel()

    point_history_classifier = PointHistoryClassifier()


    with open('type_of_model/Motionscripttype_of_model/Motionscripttype_of_model_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'type_of_model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    history_length = 16
    point_history = deque(maxlen=history_length)

    finger_gesture_history = deque(maxlen=history_length)

    #  
    type_of_mode = 0

    while True:
        fps = cvFpsCalc.get()
        
        key = cv.waitKey(10)
        if key == 27:  # ESC-----------> MotionScript Output Voice over here
            corrected_text = autocorrect_text(concatenated_hand_sign_text)
            print(corrected_text)
            
            text_to_speech(corrected_text)
            break
        
        
        number, type_of_mode = select_type_of_mode(key, type_of_mode)

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
       
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
               
                brect = calc_bounding_rect(debug_image, hand_landmarks)
             
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

              
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = point_hist(
                    debug_image, point_history)
           
                Log_into_table(number, type_of_mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

               
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                print(hand_sign_id)
                if hand_sign_id == 2:  
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

               
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

               
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
              
                debug_image = webcamscreen_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

      
        debug_image = webcamscreen(debug_image, fps, type_of_mode, number)

       
        cv.imshow('Hand Gesture Recognition', debug_image)
       
    cap.release()
    cv.destroyAllWindows()


def select_type_of_mode(key, type_of_mode):
    number = -1
    if 97 <= key <= 122: 
        number = key - 97
    if key == 32: 
        type_of_mode = 1
    
    return number, type_of_mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []


    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
       

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

 
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))


    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def point_hist(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

   
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

 
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def Log_into_table(number, type_of_mode, landmark_list, point_history_list):
    if type_of_mode == 0:
        pass
    if type_of_mode == 1 and (0 <= number <= 25):
        csv_path = 'type_of_model/Motionscripttype_of_model/Motionscripttype_of_model.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
   

#AUTOCORRECT for sign lang input feature, i have to add LLm part to it as well

import re
import string
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance
from nltk.corpus import words


correct_words = set(words.words())


def find_closest_word(word):
    min_distance = float('inf')
    closest_word = None
    for correct_word in correct_words:
        distance = edit_distance(word, correct_word)
        if distance < min_distance:
            min_distance = distance
            closest_word = correct_word
    return closest_word


def autocorrect_text(text):
    corrected_text = []
 
    words = re.findall(r'\w+|[^\w\s]', text)
    for word in words:

        if word.isalpha():
 
            if word.lower() not in correct_words:
 
                closest_word = find_closest_word(word.lower())
       
                if word[0].isupper():
                    closest_word = closest_word.capitalize()
                corrected_text.append(closest_word)
            else:
                corrected_text.append(word)
        else:
       
            corrected_text.append(word)

    return ' '.join(corrected_text)





def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 4)  

    return image


import cv2 as cv
import time

hand_sign_text = ""

last_update_time = time.time()
tab_pressed = False  


time.sleep(2)

def webcamscreen_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    global last_update_time, concatenated_hand_sign_text, tab_pressed
    
    
    current_time = time.time()
    if current_time - last_update_time >= 1.9:
        concatenated_hand_sign_text += hand_sign_text
        last_update_time = current_time

    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (255, 255, 255), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv.LINE_AA)

   

  
    cv.putText(image, "User: " + concatenated_hand_sign_text, (10, 630),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (110, 50, 50), 1, cv.LINE_AA)

  
    key = cv.waitKey(1)
    if key == 9:  # Tab key
        tab_pressed = True
    elif key == 113:  # Q key

        if concatenated_hand_sign_text:
            concatenated_hand_sign_text = concatenated_hand_sign_text[:-1]
    else:
        tab_pressed = False

   
    if tab_pressed and (len(concatenated_hand_sign_text) == 0 or concatenated_hand_sign_text[-1] != " "):
        concatenated_hand_sign_text += " "
    
    


    return image




def webcamscreen(image, fps, type_of_mode, number):
  
    cv.putText(image, "MotionScript", (10, 30), cv.FONT_HERSHEY_DUPLEX,
               1.5, (255, 255, 255), 2, cv.LINE_AA)

    
    cv.putText(image,"User: " +concatenated_hand_sign_text, (10, 200),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
    type_of_mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= type_of_mode <= 2:
        cv.putText(image, "type_of_mode:" + type_of_mode_string[type_of_mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 25:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
