import cv2
import dlib
import pyttsx3
import numpy as np
from scipy.spatial import distance as dist
import os
import winsound  # Using winsound for alarm sound on Windows
from twilio.rest import Client  # For SMS alerts
import speech_recognition as sr  # For voice commands
import sounddevice as sd  # For recording audio

# Set new alarm sound file
drowsiness_alarm = "car_horn.wav"  # Loud sound for drowsiness detection
dim_light_alarm = "beep.wav"  # Soft sound for dim light detection (ensure this file exists)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 250)  # Speed of speech
engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

# Download the required shape predictor file if not present
if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
    import urllib.request
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    print("Downloading shape predictor model...")
    urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
    import bz2
    with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2", "rb") as f_in:
        with open("shape_predictor_68_face_landmarks.dat", "wb") as f_out:
            f_out.write(f_in.read())

# Load pre-trained face and landmark detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # Vertical distance
    B = dist.euclidean(mouth[4], mouth[8])   # Vertical distance
    C = dist.euclidean(mouth[0], mouth[6])   # Horizontal distance
    mar = (A + B) / (2.0 * C)
    return mar


def adjust_brightness(frame, brightness_factor=1.5):
    """Adjust brightness of the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], int(brightness_factor * 50))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Twilio setup
account_sid = 'ACf4d08ff8097964ac855eda115e91e081'
auth_token = 'a4447b3d7ff3e990296a47cd8bfa52b2'
client = Client(account_sid, auth_token)

def send_sms_alert():
    message = client.messages.create(
        body="Drowsiness Alert! Please check on the driver.",
        from_='+16204669949',  # Your Twilio number
        to='+919685999900'
          # Emergency contact number
    )
    print(f"SMS sent: {message.sid}")

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.7
EYE_AR_CONSEC_FRAMES = 20
BLINK_RATE_THRESH = 15  # Blinks per minute
COUNTER = 0
BLINK_COUNTER = 0

cap = cv2.VideoCapture(0)
cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

DIM_LIGHT_THRESH = 50  # Threshold for dim light detection
BRIGHTNESS_ADJUSTMENT = True  # Enable brightness adjustment

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    # Dim light detection
    if avg_brightness < DIM_LIGHT_THRESH:
        cv2.putText(frame, "Low Light Detected!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        if BRIGHTNESS_ADJUSTMENT:
            frame = adjust_brightness(frame)
        try:
            winsound.PlaySound(dim_light_alarm, winsound.SND_ASYNC)
        except Exception as e:
            print("Error playing dim light sound:", e)

    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        
        leftEye = shape[36:42]
        rightEye = shape[42:48]
        mouth = shape[48:68]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Blink rate monitoring
        if ear < EYE_AR_THRESH:
            BLINK_COUNTER += 1

        # Yawning detection
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning Detected!", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        # Drowsiness detection
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                try:
                    winsound.PlaySound(drowsiness_alarm, winsound.SND_FILENAME)
                    engine.say("WAKE UP! You are feeling sleepy.")
                    engine.runAndWait()
                    send_sms_alert()
                except Exception as e:
                    print("Error playing alarm sound or voice alert:", e)
                send_sms_alert()
        else:
            COUNTER = 0
        
        for (x, y) in np.concatenate((leftEye, rightEye, mouth), axis=0):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



