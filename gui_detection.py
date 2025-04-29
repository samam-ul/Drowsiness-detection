import tkinter as tk
from threading import Thread
import cv2
from ultralytics import YOLO
import pyttsx3
import time

# Load YOLO model
model = YOLO("C:\\best.pt")

# TTS setup
tts = pyttsx3.init()
tts.setProperty('rate', 150)
last_spoken_time = 0
speak_interval = 5  # seconds

# Global variable to control webcam loop
running = False

def speak_warning(label):
    global last_spoken_time
    current_time = time.time()
    if current_time - last_spoken_time > speak_interval:
        tts.say(f"Warning! {label} detected!")
        tts.runAndWait()
        last_spoken_time = current_time

def detect():
    global running
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        original_h, original_w = frame.shape[:2]
        resized_frame = cv2.resize(frame, (416, 416))

        results = model.predict(resized_frame, imgsz=416, conf=0.25, stream=True)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names.get(cls_id, "Unknown")
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Scale coordinates
                scale_x = original_w / 416
                scale_y = original_h / 416
                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)

                # Color and warning
                if label.lower() in ["drowsy", "sleep"]:
                    color = (0, 0, 255)  # Red
                    speak_warning(label)
                else:
                    color = (0, 255, 0)  # Green

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_detection():
    global running
    running = True
    Thread(target=detect).start()

def stop_detection():
    global running
    running = False

# GUI
app = tk.Tk()
app.title("Drowsiness Detection")
app.geometry("300x150")

tk.Label(app, text="Drowsiness Detector", font=("Arial", 16)).pack(pady=10)

start_btn = tk.Button(app, text="Start Detection", font=("Arial", 12), command=start_detection)
start_btn.pack(pady=5)

stop_btn = tk.Button(app, text="Stop", font=("Arial", 12), command=stop_detection)
stop_btn.pack(pady=5)

app.mainloop()

