import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Fast train a tiny model to get the website working immediately
train_dir = 'dataset/Training'
test_dir = 'dataset/Testing'

datagen = ImageDataGenerator(rescale=1./255)

print("Loading data...")
train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Use lightweight MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training lightweight model (1 epoch)...")
model.fit(train_gen, epochs=1, steps_per_epoch=10) # train on just 10 batches to make it instant

model.save('model.h5')
print("model.h5 saved successfully!")
