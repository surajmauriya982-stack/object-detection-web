from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Electrical & Electronic Classes
ELECTRICAL_CLASSES = [
    'earbuds', 'neckband', 'earphones', 'earphone wire',
    'headphones', 'mobile charger', 'charging cable', 'wire',
    'smartwatch', 'mobile phone', 'laptop', 'power bank',
    'computer mouse', 'keyboard', 'speaker', 'remote control',
    'extension board', 'adapter'
]

# Generate distinct bright colors for each class
np.random.seed(42)
COLORS = [
    f"#{r:02x}{g:02x}{b:02x}"
    for r, g, b in np.random.randint(60, 255, size=(len(ELECTRICAL_CLASSES), 3))
]

# Load lightweight custom ONNX model into ONNX Runtime
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8s-worldv2-int8.onnx")
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 2
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(MODEL_PATH, opts, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image data received"}), 400

        # Decode base64 JPEG
        img_str = data["image"]
        if "," in img_str:
            img_str = img_str.split(",", 1)[1]

        img_bytes = base64.b64decode(img_str)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        orig_h, orig_w = frame.shape[:2]

        # Resize to YOLO input 640x640
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (640, 640))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)

        # Run ONNX inference
        preds = session.run([output_name], {input_name: input_img})[0][0]  # (22, 8400)
        preds = np.transpose(preds)  # (8400, 22)

        boxes_data = preds[:, :4]
        classes_scores = preds[:, 4:]
        class_ids = np.argmax(classes_scores, axis=1)
        confidences = np.max(classes_scores, axis=1)

        # Confidence threshold (0.28 for sensitive gadget detection like small earbuds)
        mask = confidences > 0.28
        boxes_data = boxes_data[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        boxes = []
        for box in boxes_data:
            cx, cy, bw, bh = box
            # Return normalized coordinates (0..1)
            x = (cx - bw / 2) / 640.0
            y = (cy - bh / 2) / 640.0
            w = bw / 640.0
            h = bh / 640.0
            boxes.append([x, y, w, h])

        indices = []
        if len(boxes) > 0:
            pixel_boxes = [[int(b[0]*640), int(b[1]*640), int(b[2]*640), int(b[3]*640)] for b in boxes]
            indices = cv2.dnn.NMSBoxes(pixel_boxes, confidences.tolist(), 0.28, 0.45)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                cid = int(class_ids[i])
                if cid < len(ELECTRICAL_CLASSES):
                    conf = float(confidences[i])
                    bx, by, bw, bh = boxes[i]

                    detections.append({
                        "name": ELECTRICAL_CLASSES[cid],
                        "confidence": round(conf * 100, 1),
                        "color": COLORS[cid],
                        "box": {
                            "x": max(0.0, float(bx)),
                            "y": max(0.0, float(by)),
                            "w": min(1.0, float(bw)),
                            "h": min(1.0, float(bh))
                        }
                    })

        return jsonify({
            "detections": detections,
            "count": len(detections)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
