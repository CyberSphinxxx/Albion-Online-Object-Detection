import pyautogui
import time
import random
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

# Load YOLO model
model = YOLO("runs/detect/train3/weights/best.pt")  # adjust to your path

# Define screen capture region
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # change for your screen

sct = mss()

def random_delay(base=0.2, jitter=0.3):
    time.sleep(base + random.uniform(0, jitter))

def move_mouse_smoothly(x, y):
    duration = random.uniform(0.2, 0.4)
    pyautogui.moveTo(x + random.randint(-3, 3), y + random.randint(-3, 3), duration=duration)

while True:
    # Capture screen
    img = np.array(sct.grab(monitor))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Run YOLO detection
    results = model(img_rgb)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()

        for i, box in enumerate(boxes):
            conf = confidences[i]
            class_id = int(class_ids[i])

            if conf < 0.6:
                continue  # ignore low-confidence

            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            print(f"Detected: class={class_id} at ({cx}, {cy})")

            move_mouse_smoothly(cx, cy)
            random_delay(0.1, 0.2)
            pyautogui.click()
            random_delay(0.5, 1.0)  # pause between clicks

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
