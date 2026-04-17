from flask import Flask, render_template, request
import os
import cv2

from predict import predict, preprocess_image, model
from gradcam import get_gradcam, overlay_heatmap
from simulation import simulate_risk

# ✅ CREATE APP FIRST
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"
RESULT_FOLDER = "static/results/"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    heatmap_path = None

    if request.method == "POST":
        print("✅ POST request received")

        if "file" not in request.files:
            print("❌ No file uploaded")
            return render_template("index.html")

        file = request.files["file"]

        if file.filename == "":
            print("❌ Empty filename")
            return render_template("index.html")

        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print("✅ File saved:", filepath)

        # Prediction
        label, confidence = predict(filepath)
        print("✅ Prediction:", label, confidence)

        # Grad-CAM
        img_array = preprocess_image(filepath)
        heatmap = get_gradcam(model, img_array)
        overlay = overlay_heatmap(filepath, heatmap)

        heatmap_path = os.path.join(RESULT_FOLDER, "heatmap.jpg")
        cv2.imwrite(heatmap_path, overlay)

        # Simulation
        factor = float(request.form.get("factor", 0.2))
        new_pred, change = simulate_risk(confidence, factor)

        result = {
            "label": label,
            "confidence": round(confidence, 2),
            "new_pred": round(new_pred, 2),
            "change": round(change, 2),
            "status_class": "pred-alert" if label == "PNEUMONIA" else "pred-safe",
            "image_path": filepath
        }

    return render_template("index.html", result=result, heatmap=heatmap_path)

if __name__ == "__main__":
    app.run(debug=True)