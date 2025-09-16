import tkinter as tk
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageTk
import mss

class YOLOScreenApp:
    def __init__(self, window, model_path):
        self.window = window
        self.window.title("YOLO Screen Capture Detection")
        self.window.geometry("1000x700")

        # Load YOLO model
        self.model = YOLO(model_path)

        # Tkinter video panel
        self.frame_label = tk.Label(window)
        self.frame_label.pack()

        # Dropdown for screen selection
        self.sct = mss.mss()
        self.monitors = self.sct.monitors  # list of monitors
        self.selected_monitor = tk.IntVar(value=1)  # default monitor index

        monitor_options = [f"Monitor {i}" for i in range(1, len(self.monitors))]
        self.monitor_menu = tk.OptionMenu(window, self.selected_monitor, *range(1, len(self.monitors)))
        self.monitor_menu.pack(pady=5)

        # Buttons
        tk.Button(window, text="Start Capture", command=self.start_screen).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(window, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=10, pady=10)

        self.running = False

    def start_screen(self):
        self.running = True
        self.update_frame()

    def stop(self):
        self.running = False

    def update_frame(self):
        if self.running:
            # Pick monitor based on dropdown
            monitor_index = self.selected_monitor.get()
            monitor = self.monitors[monitor_index]

            # Capture screen
            screenshot = np.array(self.sct.grab(monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            # Run YOLO detection
            results = self.model(frame, imgsz=640)
            annotated_frame = results[0].plot()

            # Convert for Tkinter
            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            self.frame_label.imgtk = imgtk
            self.frame_label.configure(image=imgtk)

            # Update every ~20ms
            self.window.after(20, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOScreenApp(root, "runs/detect/train3/weights/best.pt")  # update with your model path
    root.mainloop()
