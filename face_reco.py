import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime

# 1. Load known faces
path = r'D:\python\AI projects\Face recognization\known_faces'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(os.path.join(path, cl))
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# 2. Encode function
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# 3. Mark attendance
def markAttendance(name):
    if not os.path.isfile('Attendance.csv'):
        with open('Attendance.csv', 'w') as f:
            f.write('Name,Time\n')

    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dtString}')

# 4. Open webcam
cap = cv2.VideoCapture(0)
attendance_done = False
face_detected_time = None  
detected_name = None

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    if facesCurFrame:
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if detected_name == name:
                    # Already detecting same face
                    elapsed_time = time.time() - face_detected_time

                    if elapsed_time > 6:  # 3 seconds
                        print(f"Attendance Marked for {name}")
                        markAttendance(name)
                        attendance_done = True
                        break
                else:
                    # New face detected, start timer
                    detected_name = name
                    face_detected_time = time.time()

    else:
        detected_name = None
        face_detected_time = None

    cv2.imshow('Webcam', img)

    if attendance_done:
        print("Attendance Done, closing camera.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Manual Exit")
        break

cap.release()
cv2.destroyAllWindows()
