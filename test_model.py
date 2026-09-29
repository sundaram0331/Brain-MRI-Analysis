import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('model.h5')

# Test notumor image
img_path = 'dataset/Testing/notumor/Te-no_1.jpg'
img = Image.open(img_path).convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)

preds = model.predict(img_array, verbose=0)[0]
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
print("Raw image predictions:")
for c, p in zip(classes, preds):
    print(f"{c}: {p*100:.2f}%")

# Test with / 255.0 just in case
img_array_scaled = img_array / 255.0
preds_scaled = model.predict(img_array_scaled, verbose=0)[0]
print("\nScaled image predictions:")
for c, p in zip(classes, preds_scaled):
    print(f"{c}: {p*100:.2f}%")

# Test with BGR just in case cv2 was used in training?
# wait, ImageDataGenerator uses RGB.
import cv2
img_cv = cv2.imread(img_path)
img_cv = cv2.resize(img_cv, (224, 224))
img_cv = np.expand_dims(img_cv, axis=0)
preds_cv = model.predict(img_cv, verbose=0)[0]
print("\nOpenCV (BGR) image predictions:")
for c, p in zip(classes, preds_cv):
    print(f"{c}: {p*100:.2f}%")

# Test OpenCV RGB
img_cv_rgb = cv2.cvtColor(img_cv[0], cv2.COLOR_BGR2RGB)
img_cv_rgb = np.expand_dims(img_cv_rgb, axis=0)
preds_cv_rgb = model.predict(img_cv_rgb, verbose=0)[0]
print("\nOpenCV (RGB) image predictions:")
for c, p in zip(classes, preds_cv_rgb):
    print(f"{c}: {p*100:.2f}%")
