# 🎯 Object Detection Web App

Real-time object detection in the **browser** using **YOLOv11 Nano** and **Flask**.  
The browser captures webcam frames → sends to Flask backend → YOLO detects → annotated frame displayed back.

---

## 🚀 Deploy to Render (Free)

### Step 1 — Push to GitHub
1. Create a new repository on [github.com](https://github.com)
2. Upload all files from this folder to that repo

### Step 2 — Deploy on Render
1. Go to [render.com](https://render.com) and sign up (free)
2. Click **New → Web Service**
3. Connect your GitHub repository
4. Set these settings:

| Setting | Value |
|---------|-------|
| **Environment** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120` |
| **Instance Type** | Free |

5. Click **Deploy** — your app will be live at `https://your-app.onrender.com`

---

## 🖥️ Run Locally

```bash
# 1. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py

# 4. Open browser
# http://localhost:5000
```

---

## 📁 Project Structure

```
Object_Detection_Web/
├── app.py                  ← Flask backend (YOLO inference)
├── templates/
│   └── index.html          ← Web UI (webcam + detection display)
├── yolo11n.pt              ← YOLO model weights
├── requirements.txt
├── Procfile                ← Render deployment config
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

```
Browser Webcam
     ↓  (base64 JPEG frame)
Flask /detect endpoint
     ↓  (OpenCV decode → YOLO inference)
Annotated frame + detections
     ↓  (base64 JPEG response)
Browser Canvas Display
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 5000 | Set automatically by Render |
| `MODEL_PATH` | `yolo11n.pt` | Path to YOLO weights |

---

## 🤖 Model Info

| Property | Value |
|----------|-------|
| Model | YOLOv11 Nano |
| File | `yolo11n.pt` (~5.4 MB) |
| Classes | 80 (COCO dataset) |
| Task | Object Detection |
