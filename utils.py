from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import copy
import itertools
import mediapipe as mp
import csv

from hand_gesture_recognition import HandGestureRecognition


# Citirea etichetelor din fisierul de etichete (labels)
with open('labels.csv', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    labels = [row[0] for row in reader]

hand_gesture_recognition = HandGestureRecognition()

def calc_bounding_rect(image, landmarks):
    image_height, image_width = image.shape[:2]
    landmark_array = np.array([[int(landmark.x * image_width), int(landmark.y * image_height)] for landmark in landmarks.landmark])
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_height, image_width = image.shape[:2]
    landmark_point = [[min(int(landmark.x * image_width), image_width - 1), min(int(landmark.y * image_height), image_height - 1)] for landmark in landmarks.landmark]
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        for connection in [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 5), (5, 9), (9, 13), (13, 17)]:
            cv2.line(image, tuple(landmark_point[connection[0]]), tuple(landmark_point[connection[1]]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[connection[0]]), tuple(landmark_point[connection[1]]), (255, 255, 255), 2)
        for landmark in landmark_point:
            cv2.circle(image, tuple(landmark), 5, (255, 255, 255), -1)
            cv2.circle(image, tuple(landmark), 5, (0, 0, 0), 1)
    return image

def draw_bounding_rect(image, brect):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text += ':' + hand_sign_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def create_card1(nume, prenume, studii, absolvire,medie,olimpic,provenienta,admitere,sesiune_admitere,promotie):
    text= f"""\
Nume\t{nume}
Prenume\t{prenume}
Categorie studii\t{studii}
An absolvire liceu\t{absolvire}
Medie bacalaureat\t{medie}
Olimpic\t{olimpic}
Provenienţă\t{provenienta}
Medie admitere\t{admitere}
Sesiune admitere\t{sesiune_admitere}
Promoţie\t{promotie}
"""
    return text
def create_card2(facultate, tip_studii, specializare,fi,an,modul,grupa,matricol,status):
    text= f"""Facultate/colegiu\t{facultate}
Tip studii\t{tip_studii}
Profil\tDomeniu: {specializare}
Specializare\t{specializare}
FI\t{fi}
An studiu\t{an}
Modul\t{modul}
Grupă\t{grupa}
Nr. matricol\t{matricol}
Tip loc\tRoman bugetat
Status curent\t{status}
"""
    return text


def get_student_data():
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT an_s, an_u, sem, ponderata, aritmetica FROM your_table_name')
    data = cursor.fetchall()
    conn.close()
    return data
