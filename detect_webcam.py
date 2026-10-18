import cv2
import numpy as np
import onnxruntime as ort
import subprocess
import threading
import time

ELECTRICAL_CLASSES = [
    'earbuds', 'neckband', 'earphones', 'earphone wire',
    'headphones', 'mobile charger', 'charging cable', 'wire',
    'smartwatch', 'mobile phone', 'laptop', 'power bank',
    'computer mouse', 'keyboard', 'speaker', 'remote control',
    'extension board', 'adapter'
]

np.random.seed(42)
COLORS = [tuple(int(c) for c in color) for color in np.random.randint(60, 255, size=(len(ELECTRICAL_CLASSES), 3))]

# Non-blocking voice speaker announcement (uses Windows native speech synthesizer)
last_spoken = {}
def speak_voice(text):
    now = time.time()
    if text in last_spoken and (now - last_spoken[text] < 3.5):
        return
    last_spoken[text] = now
    def _run():
        cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
        subprocess.run(["powershell", "-NoProfile", "-Command", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    threading.Thread(target=_run, daemon=True).start()

print("Loading Electrical Gadget Detection Model...")
session = ort.InferenceSession("yolov8s-worldv2-int8.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started with Speaker Voice! Press 'Q' to exit.")
speak_voice("Camera started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = session.run([output_name], {input_name: img})[0][0]
    preds = np.transpose(preds)

    boxes_data = preds[:, :4]
    classes_scores = preds[:, 4:]
    class_ids = np.argmax(classes_scores, axis=1)
    confidences = np.max(classes_scores, axis=1)

    mask = confidences > 0.28
    boxes_data = boxes_data[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    boxes = []
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    for box in boxes_data:
        cx, cy, bw, bh = box
        x = int((cx - bw / 2) * scale_x)
        y = int((cy - bh / 2) * scale_y)
        w = int(bw * scale_x)
        h = int(bh * scale_y)
        boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences.tolist(), 0.28, 0.45)

    if len(indices) > 0:
        for i in indices.flatten():
            cid = int(class_ids[i])
            if cid < len(ELECTRICAL_CLASSES):
                x, y, w, h = boxes[i]
                cname = ELECTRICAL_CLASSES[cid]
                conf = float(confidences[i])
                color = COLORS[cid]

                # Draw box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

                # Draw prominent label DIRECTLY ABOVE the box
                label = f"{cname.upper()} {round(conf * 100)}%"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                badge_y = max(0, y - lh - 10)
                cv2.rectangle(frame, (x, badge_y), (x + lw + 10, y), color, -1)
                cv2.putText(frame, label, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # Speak the object through the speaker!
                speak_voice(cname)

    cv2.imshow("Electrical Gadget Detection + Voice Speaker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
