import cv2
import numpy as np
import time
import sounddevice as sd
import threading
import speech_recognition as sr
import tkinter as tk
from tkinter import messagebox


stop_threads = False 
# ======================
# SOUND DETECTION MODE
def speech_to_text_with_video():
    global stop_threads
    stop_threads = False
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    recognized_text = {"text": ""}

    def audio_listener():
        global stop_threads
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)  # better accuracy

        while not stop_threads:
            try:
                with mic as source:
                    print("Listening...")
                    audio = recognizer.listen(source, phrase_time_limit=4)  # capture 4-sec chunks
                text = recognizer.recognize_google(audio, language="en-IN")  # use Indian English
                recognized_text["text"] = text
                print("Recognized:", text)
            except sr.UnknownValueError:
                recognized_text["text"] = ""
            except sr.RequestError as e:
                recognized_text["text"] = f"[API Error: {e}]"

    # Run mic listener in background
    threading.Thread(target=audio_listener, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw recognized speech on video
        if recognized_text["text"]:
            cv2.putText(frame, recognized_text["text"], (40, 80),
                        font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "press [esc] to exit", (40, 460),
                    font, 0.6, (0, 255, 255), 2)

        cv2.imshow("Speech Detection", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            stop_threads = True
            break

    cap.release()
    cv2.destroyAllWindows()
    open_gui()


# ======================
# OBJECT DETECTION MODE
# ======================
def object_detection():
    messagebox.showinfo("Mode Selected", "Running Object Detection Mode")

    # Load YOLO model
    net = cv2.dnn.readNet("./weights/yolov3-tiny.weights", "./configuration/yolov3-tiny.cfg")
    with open("./configuration/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Open webcam
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    starting_time = time.time()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        height, width, channels = frame.shape

        # --- Preprocess frame for YOLO ---
        # Resize with unchanged aspect ratio using padding (letterbox)
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416),
                                     (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # --- Process detections ---
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:  # threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {round(confidence, 2)}",
                            (x, y - 10), font, 0.7, color, 2)

        # Show FPS
        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, f"FPS: {round(fps, 2)}", (20, 40),
                    font, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, "press [esc] to exit", (20, height - 20),
                    font, 0.6, (0, 255, 255), 2)

        cv2.imshow("Object Detection", frame)

        # Exit on ESC
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    open_gui()

# ======================
# GUI for Mode Selection
# ======================
def open_gui():
    root = tk.Tk()
    root.title("Detection Mode Selection")

    # Fullscreen mode
    root.attributes("-fullscreen", True)
    root.config(bg="#121212")

    # Exit fullscreen with Esc
    root.bind("<Escape>", lambda event: root.destroy())

    # Title
    title = tk.Label(root, text="🔍 RealScan", font=("Arial", 40, "bold"),
                     bg="#121212", fg="white")
    title.pack(pady=40)

    subtitle = tk.Label(root, text="Choose a detection mode to continue",
                        font=("Arial", 20), bg="#121212", fg="#bbbbbb")
    subtitle.pack(pady=10)

    # Hover effect function
    def on_enter(e):
        e.widget['background'] = "#333333"
    def on_leave(e):
        e.widget['background'] = e.widget.default_bg

    # Custom button creator
    def create_button(text, color, command):
        btn = tk.Button(root, text=text, font=("Arial", 22, "bold"),
                        bg=color, fg="white", activebackground="#555555",
                        relief="flat", width=25, height=2,
                        command=command)
        btn.default_bg = color
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    # Buttons
    btn1 = create_button("▶ Object Detection", "#4CAF50",
                         lambda: [root.destroy(), object_detection()])
    btn1.pack(pady=20)

    btn2 = create_button("🎤 Sound Detection", "#2196F3",
                         lambda: [root.destroy(), speech_to_text_with_video()])
    btn2.pack(pady=20)

    btn3 = create_button("❌ Exit", "red", root.destroy)
    btn3.pack(pady=40)

    # Exit confirmation (Alt+F4 or close button)
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to exit?"):
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()



# Run GUI
open_gui()
