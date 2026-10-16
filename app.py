from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Load YOLO model once at startup (relative path — portable)
MODEL_PATH = os.environ.get("MODEL_PATH", "yolo11n.pt")
model = YOLO(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data received"}), 400

        # Strip base64 header (e.g. "data:image/jpeg;base64,")
        img_data = data["image"].split(",")[1]

        # Decode base64 -> numpy array -> OpenCV frame
        img_bytes  = base64.b64decode(img_data)
        img_array  = np.frombuffer(img_bytes, dtype=np.uint8)
        frame      = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Run YOLO inference
        results         = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Collect detected object names + confidence
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "name":       model.names[int(box.cls)],
                    "confidence": round(float(box.conf) * 100, 1)
                })

        # Encode annotated frame back to base64 JPEG
        _, buffer   = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64  = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "image":      f"data:image/jpeg;base64,{img_base64}",
            "detections": detections,
            "count":      len(detections)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
