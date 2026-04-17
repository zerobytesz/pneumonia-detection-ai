import tensorflow as tf
import numpy as np
import cv2

def get_gradcam(model, img_array):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("conv5_block16_concat").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # This line turns the tensor into a NumPy array...
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    # ...so we just return it as-is! (Removed the .numpy())
    return heatmap

def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed