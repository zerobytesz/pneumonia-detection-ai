import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3   # 🔥 keep 3 for fast + optimal training

train_dir = "chest_xray/train"
test_dir = "chest_xray/test"

os.makedirs("model", exist_ok=True)

# ==============================
# DATA GENERATORS
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ==============================
# CLASS WEIGHTS (IMPORTANT)
# ==============================
class_counts = np.bincount(train_gen.classes)
total = sum(class_counts)

class_weight = {
    0: total / class_counts[0],
    1: total / class_counts[1]
}

print("Class Weights:", class_weight)

# ==============================
# MODEL (DenseNet121)
# ==============================
base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze most layers (Phase 1)
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Custom head (as per paper)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# ==============================
# COMPILE
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# ==============================
# CALLBACKS
# ==============================
early_stop = EarlyStopping(
    monitor="val_auc",
    patience=3,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "model/model.h5",
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1
)

# ==============================
# TRAIN
# ==============================
print("\n🚀 Starting Training...\n")

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

print("\n✅ Training Complete! Best model saved at: model/model.h5")