# YOLO Screen Capture with Tkinter  

This project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for real-time **object detection** on your screen.  
It allows you to:  
✅ Select which monitor to capture  
✅ Run YOLO detections on the screen feed  
✅ Display annotated results inside a Tkinter GUI  

---

## 📌 Features
- Screen capture with [mss](https://github.com/BoboTiG/python-mss)  
- Multi-monitor support (choose which screen to capture)  
- Live YOLO detection results displayed in a Tkinter window  
- Start/Stop detection with GUI buttons  

---

## 🛠️ Installation  

1. **Clone this repository**  
```bash
git clone https://github.com/your-username/yolo-screen-tkinter.git
cd yolo-screen-tkinter
```

2. **Install dependencies**  
```bash
pip install ultralytics opencv-python pillow mss
```

3. **Train or download a YOLO model**  
- You can use a pretrained YOLO model (e.g., `yolov8n.pt`)  
- Or place your trained model (e.g., `runs/detect/train3/weights/best.pt`) inside the project folder  

---

## ▶️ Usage  

Run the app with:  
```bash
python app.py
```

### Steps inside the GUI:
1. Select which monitor you want to capture (Monitor 1, Monitor 2, etc.)  
2. Click **Start Capture** to begin detection  
3. Click **Stop** to stop detection  

---

## 📂 Project Structure  

```
yolo-screen-tkinter/
│── app.py                # Main Tkinter GUI application
│── runs/                 # (Optional) Folder for YOLO training outputs
│── README.md             # Project documentation
│── requirements.txt      # Dependencies (optional)
```

---

## ⚡ Notes
- `Monitor 1` is usually your **primary display**  
- `Monitor 2` and others will appear if you have multiple screens connected  
- Use a **GPU** for faster detection (recommended)  

---

## 📜 License
This project is open-source under the MIT License.  
