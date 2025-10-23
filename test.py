import tkinter as tk
from tkinter import messagebox
import threading
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import speech_recognition as sr
import tensorflow as tf
from tensorflow import keras


# Function to start hand detection (Sign Language Recognition)
def start_hand_detection():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier(
        r"C:\Users\bhavi\OneDrive\Desktop\Modell\keras_model.h5",
        r"C:\Users\bhavi\OneDrive\Desktop\Modell\labels.txt"
    )
    
    offset = 20
    imgSize = 300
    labels = ["Hello", "Please", "Thank you", "Yes"]
    
    while True:
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                
                cv2.rectangle(imgOutput, (x1, y1 - 70), (x1 + 200, y1 - 30), (255, 255, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x1 + 10, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 255, 255), 4)

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)

        cv2.imshow('Image', imgOutput)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to start speech recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        status_label.config(text="Listening... Speak now.")
        window.update_idletasks()
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            text_entry.delete(0, tk.END)
            text_entry.insert(0, text)
            status_label.config(text="Speech recognition complete.")
        except sr.UnknownValueError:
            messagebox.showerror("Error", "Could not understand the audio.")
        except sr.RequestError:
            messagebox.showerror("Error", "Check your internet connection.")
        except sr.WaitTimeoutError:
            messagebox.showwarning("Timeout", "Listening timed out. Speak within 5 seconds.")

# GUI Setup
window = tk.Tk()
window.title("Hand & Speech Recognition")
window.geometry("960x600")
window.configure(bg="#AFEEEE")

frame = tk.Frame(window, bg="#FFDAB9")
frame.pack(pady=40)

title_label = tk.Label(frame, text="Select an Option", font=("Algerian", 30, "bold"), bg="#F4F7FC", fg="#000000")
title_label.pack(pady=10)

choice_label = tk.Label(frame, text="Choose an option:", font=("Copperplate Gothic Bold", 20), bg="#F4F7FC", fg="#000000")
choice_label.pack(pady=5)

choice_entry = tk.Entry(frame, font=("Copperplate Gothic Bold", 18), bg="#EAEAEA", fg="black", bd=2, relief="solid", width=10)
choice_entry.pack(pady=15)

def option_1():
    threading.Thread(target=start_hand_detection, daemon=True).start()

def option_2():
    threading.Thread(target=recognize_speech, daemon=True).start()

option_1_button = tk.Button(frame, text="Open Camera for Hand Detection", command=option_1, font=("Copperplate Gothic Bold", 16), bg="#F1A7B8", fg="Green", relief="flat", padx=30, pady=10)
option_1_button.pack(pady=5)

option_2_button = tk.Button(frame, text="Voice to Text", command=option_2, font=("Copperplate Gothic Bold", 16), bg="#FFD700", fg="Blue", relief="flat", padx=30, pady=10)
option_2_button.pack(pady=5)

submit_button = tk.Button(frame, text="Submit", font=("Copperplate Gothic Bold", 18), bg="#9D5FF0", fg="white", relief="flat", padx=40, pady=10)
submit_button.pack(pady=20)

instructions_label = tk.Label(frame, text="Type '1' for Hand Detection or '2' for Speech Recognition.", font=("Copperplate Gothic Bold", 14), bg="#F4F7FC", fg="#000000")
instructions_label.pack(pady=10)

def submit_button_click():
    user_input = choice_entry.get()
    if user_input == "1":
        option_1()
    elif user_input == "2":
        option_2()
    else:
        messagebox.showerror("Error", "Invalid input. Enter 1 or 2.")

submit_button.config(command=submit_button_click)

text_entry = tk.Entry(window, font=("Helvetica", 24), width=36)
text_entry.pack(pady=10)

status_label = tk.Label(window, text="Press 'Voice to Text' and speak.", font=("Helvetica", 14), bg="#AFEEEE", fg="black")
status_label.pack(pady=10)

window.mainloop()
