from flask import Flask, request, jsonify, render_template, session, flash
from flask_session import Session
import cv2
import numpy as np
import base64
import copy
import itertools
import mediapipe as mp
import csv

from utils import calc_bounding_rect, calc_landmark_list,pre_process_landmark,draw_landmarks,draw_bounding_rect,draw_info_text,create_card1,create_card2


from hand_gesture_recognition import HandGestureRecognition
import mysql.connector

app = Flask(__name__)
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


app.secret_key="aceasta_este_o_cheie_secreta_aparent"
app.config['SESSION_TYPE'] = 'filesystem'  # Poți alege și alte opțiuni, cum ar fi 'redis', 'mongodb', etc.
Session(app)

with open('labels.csv', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    labels = [row[0] for row in reader]

hand_gesture_recognition = HandGestureRecognition()
connection=mysql.connector.connect(host="localhost", user="root", password="", database="licenta")

if connection.is_connected():
        print("Connected Successfully")
else:
        print("Fail to connect")

connection.close()

connection=mysql.connector.connect(host="localhost", user="root", password="", database="licenta")
cursor= connection.cursor()


@app.route('/process_frame', methods=['POST'])
def process_frame():
    frame_data = request.get_json().get('frameData')
    image_data = base64.b64decode(frame_data.split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    index_finger_tipx = None
    index_finger_tipy = None
    gesture = None

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            brect = calc_bounding_rect(frame, hand_landmarks)
            landmark_list = calc_landmark_list(frame, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            hand_sign_id = hand_gesture_recognition(pre_processed_landmark_list)
            hand_sign_text = labels[hand_sign_id]

            frame = draw_bounding_rect(frame, brect)
            frame = draw_landmarks(frame, landmark_list)
            frame = draw_info_text(frame, brect, handedness, hand_sign_text)

            # Coordonatele vârfului degetului arătător
            index_finger_tipx = (landmark_list[8][0])
            index_finger_tipy = ( landmark_list[8][1])
            # Setarea gestului recunoscut
            gesture = hand_sign_text
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'frameData': 'data:image/jpeg;base64,' + encoded_frame,
        'indexFingerTipx': index_finger_tipx,
        'indexFingerTipy': index_finger_tipy,
        'gesture': gesture
    })

@app.route('/',methods=['POST', 'GET'])
def index():
    return render_template('login.html')


@app.route('/index',methods=['POST', 'GET'])
def index1():
    query = """SELECT prenume, nume, email, studii, absolvire, medie, olimpic, provenienta, admitere, sesiune_admitere, promotie,
                      facultate, tip_studii, specializare, fi, an, modul, grupa, matricol, status 
               FROM user 
               WHERE email = %s"""
    email = session.get('email')
    cursor.execute(query, (email,))
    result = cursor.fetchone() 
    
    # Verifică dacă rezultatul nu este None
    if result:
        # Primele 10 valori pentru create_card1
        prenume, nume, email, studii, absolvire, medie_bac, olimpic, provenienta, admitere, sesiune_admitere,promotie = result[:11]
        if(olimpic==True):
             olimpic='DA'
        else: olimpic='NU'
        # Următoarele 9 valori pentru create_card2
        facultate, tip_studii, specializare, fi, an, modul, grupa, matricol, status = result[11:20]

        # Apelează create_card1 și create_card2 cu datele corespunzătoare
        text_card1 = create_card1(nume, prenume, studii, absolvire, medie_bac, olimpic, provenienta, admitere, sesiune_admitere,promotie)
        text_card2 = create_card2(facultate, tip_studii, specializare, fi, an, modul, grupa, matricol, status)
        # Flash mesajele pentru carduri
        flash(text_card1, 'card1')
        flash(text_card2, 'card2')
    else:
        flash("Nu s-au găsit date pentru emailul furnizat.", 'error')

    return render_template('index.html')


@app.route('/istoric_scolar' , methods=['POST', 'GET'])
def istoric():
    query = """SELECT an_s, an_u, sem, ponderata, aritmetica 
               FROM medi_final
               WHERE email = %s"""
    email = session.get('email')
    cursor.execute(query, (email,))
    data = cursor.fetchall()
    

    return render_template('istoric_scolar.html', data=data)


@app.route('/select_year', methods=['POST'])
def select_year():
    selected_year=1
    selected_year = request.form.get('year')
    query = """SELECT materie, semestru, nota, credite 
               FROM medi
               WHERE email = %s AND an= %s"""
    email = session.get('email')
    cursor.execute(query, (email,selected_year))
    data = cursor.fetchall()
    return render_template('note.html', data=data)


@app.route('/note' , methods=['POST', 'GET'])
def note():
    selected_year=1
    query = """SELECT materie, semestru, nota, credite 
               FROM medi
               WHERE email = %s AND an= %s"""
    email = session.get('email')
    cursor.execute(query, (email,selected_year))
    data = cursor.fetchall()
    return render_template('note.html', data=data)

@app.route('/login' , methods=['POST', 'GET'])
def logc():
      email='asdfguioipiuytrdfghjklkkjhg'
      email=str(request.form['username'])
      parola=str(request.form['password'])
      query = "SELECT COUNT(*) FROM user WHERE email = %s AND parola = %s"
      cursor.execute(query, (email, parola,))
      result = cursor.fetchone()
      email_exists = result[0] > 0
      
      if(email_exists==1):
            session['email'] =email
            query = """SELECT prenume, nume, email, studii, absolvire, medie, olimpic, provenienta, admitere, sesiune_admitere, promotie,
                        facultate, tip_studii, specializare, fi, an, modul, grupa, matricol, status 
                    FROM user 
                    WHERE email = %s"""
            email = session.get('email')
            cursor.execute(query, (email,))
            result = cursor.fetchone() 
            
            # Verifică dacă rezultatul nu este None
            if result:
                # Primele 10 valori pentru create_card1
                prenume, nume, email, studii, absolvire, medie_bac, olimpic, provenienta, admitere, sesiune_admitere,promotie = result[:11]
                if(olimpic==True):
                    olimpic='DA'
                else: olimpic='NU'
                # Următoarele 9 valori pentru create_card2
                facultate, tip_studii, specializare, fi, an, modul, grupa, matricol, status = result[11:20]

                # Apelează create_card1 și create_card2 cu datele corespunzătoare
                text_card1 = create_card1(nume, prenume, studii, absolvire, medie_bac, olimpic, provenienta, admitere, sesiune_admitere,promotie)
                text_card2 = create_card2(facultate, tip_studii, specializare, fi, an, modul, grupa, matricol, status)
                # Flash mesajele pentru carduri
                flash(text_card1, 'card1')
                flash(text_card2, 'card2')
            else:
                flash("Nu s-au găsit date pentru emailul furnizat.", 'error')

            return render_template('index.html')
      else:
            flash("Contul nu exista.")
            return render_template('login.html') 



if __name__ == '__main__':
    app.run(debug=True)
