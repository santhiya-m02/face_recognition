import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import joblib

app = Flask(__name__)

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if not os.path.exists('model'):
    os.makedirs('model')
   
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')
def totalreg():
    return len(os.listdir('static/faces'))

#### extract the face from an image
#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    if len(face_points) > 0:
        return face_points
    else:
        return []

def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Change the number of units in the output layer to 1 for binary classification
    model.add(Dense(1, activation='sigmoid'))

    return model


def identify_face(facearray, model):
    predictions = model.predict(np.expand_dims(facearray, axis=0))

    if predictions.size == 0:
        # Handle the case where predictions is empty
        return None, None

    label_index = np.argmax(predictions)

    if label_index >= len(predictions[0]):
        # Handle the case where label_index exceeds the length of predictions[0]
        return None, None

    confidence = predictions[0][label_index]
    return label_index, confidence


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face)
            labels.append(int(user.split('_')[1]))

    faces = np.array(faces) / 255.0
    labels = np.array(labels)

    num_classes = len(np.unique(labels))
    model = build_cnn_model(input_shape=(50, 50, 3), num_classes=num_classes)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(faces, labels, epochs=10)

    # Save the model architecture to a JSON file
    model_json = model.to_json()
    with open('model/cnn_fr.json', 'w') as json_file:
        json_file.write(model_json)

    # Save the model weights
    model.save_weights('model/weights.h5')


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    
    # Check if 'Accuracy' column exists in the dataframe
    if 'Accuracy' in df.columns:
        accuracy = df['Accuracy']
    else:
        accuracy = pd.Series([0.0] * len(df), name='Accuracy')

    l = len(df)
    return names, rolls, times, accuracy, l


def add_attendance(name, confidence):
    if isinstance(name, tuple):
        recognized_person, _ = name  # Unpack the tuple
    else:
        recognized_person = name

    # Check if recognized_person is not None and is an iterable (a tuple)
    if recognized_person is not None and hasattr(recognized_person, '__iter__') and '_' in recognized_person:
        recognized_person = str(recognized_person)  # Convert to string
        username, _, userid = recognized_person.partition('_')  # Use partition to split once

        # Check if userid is not empty
        if userid:
            current_time = datetime.now().strftime("%H:%M:%S")

            df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
            if int(userid) not in list(df['Roll']):
                with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                    f.write(f'\n{username},{userid},{current_time},{confidence}')



@app.route('/')
def home():
    names,rolls,times,Accuracy,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,Accuracy=Accuracy,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


@app.route('/start', methods=['GET'])
def start():
    model = build_cnn_model(input_shape=(50, 50, 3), num_classes=totalreg())
    model.load_weights('model/weights.h5')

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        faces = face_detector.detectMultiScale(frame, 1.3, 5)

        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            recognized_person, confidence = identify_face(face, model)

            if recognized_person is not None:
                add_attendance(recognized_person, confidence)

                name_to_display = f'{recognized_person} ({confidence:.2f})'
                cv2.putText(frame, name_to_display, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)



        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Extract attendance after the loop is completed
    names, rolls, times, Accuracy, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, Accuracy=Accuracy, l=l, totalreg=totalreg(), datetoday2=datetoday2)





#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/100',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%5==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,Accuracy,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
