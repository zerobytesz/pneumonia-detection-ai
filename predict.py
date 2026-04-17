import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

IMG_SIZE = 224

# ✅ LOAD YOUR TRAINED MODEL
model = load_model("model/model.h5")

# ==============================
# PREPROCESS IMAGE
# ==============================
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==============================
# PREDICT FUNCTION
# ==============================
def predict(img_path):
    img = preprocess_image(img_path)
    pred = model.predict(img)[0][0]

    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"

    return label, float(pred)