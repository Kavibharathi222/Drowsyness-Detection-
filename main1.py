import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pyttsx3
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
import dlib

model = load_model("Drowsy3.keras")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
##t=2
##cam.set(cv2.CAP-PROP_FRAME_WIDTH,640)
face_detector = dlib.get_frontal_face_detector()
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1.5)  # Volume level
engine.setProperty('voice', 'english-us')  # Voice
text = "Wake up dude"
dlib_facelandmark = dlib.shape_predictor("D://Shapepredictor//shape_predictor_68_face_landmarks.dat")
def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A+poi_B)/(2*poi_C)
    return aspect_ratio_Eye

while True:
    ret, img = cam.read()
    #
##    time.sleep(0.2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 1)
    faces1 = face_detector(gray)
    for face in faces1:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = [] 
        rightEye = []

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y= face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
##            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y

##        right_Eye = Detect_Eye(rightEye)
##        left_Eye = Detect_Eye(leftEye)
##        Eye_Rat = (left_Eye+right_Eye)/2
##
##        Eye_Rat = round(Eye_Rat, 2)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                
                # Resize and preprocess the eye ROI
                final_image = cv2.resize(eye_roi, (224, 224))
    ##            final_image = np.expand_dims(final_image, axis=0)
                final_image = np.stack((final_image,) * 3, axis=-1)# Add channel dimension
                final_image = np.expand_dims(final_image, axis=0)   # Add batch dimension
                final_image = final_image / 255.0                   # Normalize pixel values
                
                # Predict drowsiness using the model
                predictions = model.predict(final_image)
                
                font = cv2.FONT_HERSHEY_DUPLEX
                l=[]
                right_Eye = Detect_Eye(rightEye)
                left_Eye = Detect_Eye(leftEye)
                Eye_Rat = (left_Eye+right_Eye)/2

                Eye_Rat = round(Eye_Rat, 2)
                if predictions < 1 and Eye_Rat < 0.25:
                    #t-=1
                    #if t < 0:
                    cv2.putText(img, "Close", (50, 50), font, 3, (200, 100, 200), 2, cv2.LINE_4)
                    print("Drowsiness detection")
                    cv2.putText(img, "Alert!!!! WAKE up dude", (50, 450),cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)

                    print(predictions)
##                    plt.imshow(final_image[0])  # Display the first image in the batch
##                    plt.axis('off')  # Turn off axis
##                    plt.show()
                    engine.say(text)
                    engine.runAndWait()
##                    t+=1
    ##                else:
    ##                    pass
                else:
                    cv2.putText(img, "open ", (50, 50), font, 3, (0, 0, 200), 2, cv2.LINE_4)
                    print(predictions)
    ##                plt.imshow(final_image[0])  # Display the first image in the batch
    ##                plt.axis('off')  # Turn off axis
    ##                plt.show()
    ##                engine.say(text)
    ##                engine.runAndWait()
                    print(" No Drowsiness detection")

    cv2.imshow("Image", img)

    q = cv2.waitKey(1)
    if q == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
