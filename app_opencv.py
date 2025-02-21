import csv
import copy
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
from hand_gesture_recognition import HandGestureRecognition

def main():
    #Pregatirea capturii video ------------------------------------------
    cap_device = 0  
    cap_width = 960  
    cap_height = 540 

    use_static_image_mode = False  
    #Praguri minime de confidenta pentru detectie -------------------------------------
    min_detection_confidence = 0.7  
    min_tracking_confidence = 0.5  

    use_brect = True  

    # Pregatirea camerei ---------------------------------
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Incarcarea modelului mediapipe pentru Hand Reecognition ------------------------------
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    #definim un obiect de tipul HandGestureRecognition
    hand_gesture_recognition = HandGestureRecognition()

    # Citirea etichetelor din fisierul de etichete (labels) ---------------------------
    with open('labels.csv',
              encoding='utf-8-sig') as f:
        labels = csv.reader(f)
        labels = [
            row[0] for row in labels
        ]

    

    # Setam modul aplicatiei-------------------------
    ''' modul se refera la faptul ca aplicatia poate sau nu sa primeasca noi labels si inputuri
    mode = 1 inseamna ca putem pune inputuri noi
    '''
    mode = 0

    while True:
        # Asteptam pentru o tasta, la fiecare 10 secunde, daca tasta este ESC se iese din functie
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        #Selectam modul si sau numarul care e returnat de functia selet_mode
        number, mode = select_mode(key, mode)

        # Cream o captura de imagine de la camera
        ret, image = cap.read()
        if not ret:
            break

        #inversam imaginea orizontal
        image = cv.flip(image, 1)  

        #creem o copie a imaginii pentru a lucra pe aceasa, dorimsa prpezervam astfel imaginea initiala
        debug_image = copy.deepcopy(image)

        # Implementam detectarea ------------------------------------------------------------------------------
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False #setam flagul pe False pentru a nu modifica imaginea
        results = hands.process(image) #aplicam modelul de detectie a maini
        image.flags.writeable = True #setam flagul pe True pentru a modifica imagina ulterior

         # verificam daca au fost detectate 
        if results.multi_hand_landmarks is not None:
            #daca au fost detectate iteram pentru fiecare mana toate nodurile
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, #nodurile
                                                  results.multi_handedness): # daca o mana e dreapta sau stanga
                # Calculam marginile chenarului din jurul mainii
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # calculam lista
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list)

                # Hand sign classification
                hand_sign_id = hand_gesture_recognition(pre_processed_landmark_list)

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    labels[hand_sign_id],
                )

       # debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


#Definim functia pentru selectare a modului----------------------------------------------------------------
def select_mode(key, mode):
    number = -1 
    # Verificăm dacă tasta apăsată este un număr între 0 și 9
    if ord('0') <= key <= ord('9'):
        # Convertim codul ASCII al cifrei în numărul corespunzător
        number = key - ord('0')
    
    # Verificăm dacă tasta apăsată este tasta 'k' (ASCII 107)
    if key == 107:
        # Schimbăm modul aplicației la 1
        mode = 1
    return number, mode

#Definim functia pentru a crea chenarul din jurul mainii ------------------------------------------------------------------
def calc_bounding_rect(image, landmarks):
    image_height, image_width = image.shape[:2]

    # extragem coorinatele nodurilor de pe m ana
    landmark_array = np.array([[int(landmark.x * image_width), 
                                int(landmark.y * image_height)] 
                                for landmark in landmarks.landmark])

    # calculam dimensiunile triunghiului
    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


#Definim functia care calculeaza coordonatele nodurilor de pe mana si le adauga intr-o lista --------------------------------------------------------------------
def calc_landmark_list(image, landmarks):
    image_height, image_width = image.shape[:2]

    landmark_point = [[min(int(landmark.x * image_width), image_width - 1),
                       min(int(landmark.y * image_height), image_height - 1)]
                      for landmark in landmarks.landmark]

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'x_y_values.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Draw hand connections
        for connection in [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
                           (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
                           (17, 18), (18, 19), (19, 20), (0, 5), (5, 9), (9, 13), (13, 17)]:
            cv.line(image, tuple(landmark_point[connection[0]]), tuple(landmark_point[connection[1]]),
                     (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[connection[0]]), tuple(landmark_point[connection[1]]),
                     (255, 255, 255), 2)

    # Draw hand landmarks
    for landmark in landmark_point:
        cv.circle(image, tuple(landmark), 5, (255, 255, 255), -1)
        cv.circle(image, tuple(landmark), 5, (0, 0, 0), 1)

    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_info(image, mode, number):
   
    mode_string = ['Logging Key Point']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
