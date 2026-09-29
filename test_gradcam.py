import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm

model = tf.keras.models.load_model('model.h5')

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
        raise ValueError("Gradients are None!")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

img_path = 'dataset/Testing/notumor/Te-no_1.jpg'
img = Image.open(img_path).convert('RGB')
img_resized = img.resize((224, 224))
img_array = np.array(img_resized) / 255.0
img_array = np.expand_dims(img_array, axis=0)

try:
    preds = model.predict(img_array)[0]
    pred_idx = int(np.argmax(preds))
    heatmap = make_gradcam_heatmap(img_array, model, 'out_relu', pred_idx)
    print("Heatmap generated successfully!")
    print("Heatmap shape:", heatmap.shape)
except Exception as e:
    import traceback
    traceback.print_exc()

def display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.width, img.height))
    
    superimposed_img = Image.blend(img.convert("RGB"), jet_heatmap.convert("RGB"), alpha)
    return superimposed_img

try:
    overlay = display_gradcam(img, heatmap)
    print("Overlay created!")
except Exception as e:
    import traceback
    traceback.print_exc()
