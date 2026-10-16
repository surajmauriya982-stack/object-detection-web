from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Generate distinct colors for classes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)

# Load lightweight ONNX model into ONNX Runtime (uses only ~60MB RAM)
MODEL_PATH = os.environ.get("MODEL_PATH", "yolo11n.onnx")
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 2
session = ort.InferenceSession(MODEL_PATH, opts, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data received"}), 400

        # Strip base64 header
        img_data = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        orig_h, orig_w = frame.shape[:2]

        # Preprocess frame for YOLO (640x640, RGB, normalized)
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (640, 640))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)

        # Run ONNX inference
        preds = session.run([output_name], {input_name: input_img})[0][0]  # (84, 8400)
        preds = np.transpose(preds)  # (8400, 84)

        boxes_data = preds[:, :4]
        classes_scores = preds[:, 4:]
        class_ids = np.argmax(classes_scores, axis=1)
        confidences = np.max(classes_scores, axis=1)

        # Confidence filter
        mask = confidences > 0.35
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

        indices = cv2.dnn.NMSBoxes(boxes, confidences.tolist(), 0.35, 0.45)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cid = int(class_ids[i])
                cname = COCO_CLASSES[cid]
                conf = float(confidences[i])
                color = [int(c) for c in COLORS[cid]]

                detections.append({
                    "name": cname,
                    "confidence": round(conf * 100, 1)
                })

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Draw label badge
                label = f"{cname} {round(conf * 100)}%"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, max(0, y - lh - 6)), (x + lw + 4, max(lh + 6, y)), color, -1)
                cv2.putText(frame, label, (x + 2, max(lh + 2, y - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Encode frame back to JPEG base64
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "image": f"data:image/jpeg;base64,{img_base64}",
            "detections": detections,
            "count": len(detections)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
