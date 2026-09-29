"""
NeonCortrex-CNN: Custom Brain Tumor Classifier
A completely original CNN architecture with:
  - Custom Squeeze-and-Excitation (SE) Attention blocks
  - Residual skip connections
  - SpatialDropout2D for MRI noise robustness
  - Trained from scratch — no transfer learning
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 40
LR          = 0.001
TRAIN_DIR   = '/Users/meenakshisundaram/Documents/Antigravity/Brain-Tumor-Futuristic/dataset/Training'
TEST_DIR    = '/Users/meenakshisundaram/Documents/Antigravity/Brain-Tumor-Futuristic/dataset/Testing'
MODEL_SAVE  = '/Users/meenakshisundaram/Documents/Antigravity/Brain-Tumor-Futuristic/neoncortrex_cnn.keras'
CLASSES     = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def make_dataframes(train_dir, test_dir):
    def load_dir(directory):
        fps, labels = [], []
        for cls in sorted(os.listdir(directory)):
            cls_path = os.path.join(directory, cls)
            if os.path.isdir(cls_path):
                for f in os.listdir(cls_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        fps.append(os.path.join(cls_path, f))
                        labels.append(cls)
        return pd.DataFrame({'filepaths': fps, 'labels': labels})

    train_full = load_dir(train_dir)
    test_df    = load_dir(test_dir)
    train_df, val_df = train_test_split(
        train_full, test_size=0.15, random_state=42, stratify=train_full['labels']
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df

train_df, val_df, test_df = make_dataframes(TRAIN_DIR, TEST_DIR)

# ─── GENERATORS ───────────────────────────────────────────────────────────────
train_gen_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.85, 1.15],
)
val_gen_aug = ImageDataGenerator(rescale=1./255)

train_flow = train_gen_aug.flow_from_dataframe(
    train_df, x_col='filepaths', y_col='labels',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True
)
val_flow = val_gen_aug.flow_from_dataframe(
    val_df, x_col='filepaths', y_col='labels',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)
test_flow = val_gen_aug.flow_from_dataframe(
    test_df, x_col='filepaths', y_col='labels',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

# ─── CUSTOM BUILDING BLOCKS ───────────────────────────────────────────────────

def squeeze_excitation_block(x, ratio=16):
    """
    Squeeze-and-Excitation (SE) Attention Block.
    Novel addition: learns WHICH feature channels are most important
    for distinguishing brain tumor types.
    """
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(se)
    se = layers.Dense(filters, activation='sigmoid',
                      kernel_regularizer=regularizers.l2(1e-4))(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def conv_block(x, filters, kernel_size=3, stride=1, use_bn=True):
    """Standard conv → BN → ReLU block."""
    x = layers.Conv2D(
        filters, kernel_size, strides=stride,
        padding='same', use_bias=not use_bn,
        kernel_regularizer=regularizers.l2(1e-4),
        kernel_initializer='he_normal'
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def neoncortrex_block(x, filters, dropout_rate=0.2):
    """
    NeonCortrex custom block:
    Double Conv → SE Attention → Residual Skip → SpatialDropout
    """
    # Main path: 2 convolutions
    out = conv_block(x, filters)
    out = conv_block(out, filters)

    # Squeeze-and-Excitation channel attention
    out = squeeze_excitation_block(out)

    # Residual skip connection (match channels with 1×1 conv if needed)
    if x.shape[-1] != filters:
        x = layers.Conv2D(filters, 1, padding='same',
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
    out = layers.Add()([out, x])
    out = layers.ReLU()(out)

    # SpatialDropout: drops entire feature maps (more robust than standard dropout)
    out = layers.SpatialDropout2D(dropout_rate)(out)

    # Max pooling to downsample
    out = layers.MaxPooling2D(2)(out)
    return out


# ─── NEONCORTREX-CNN ARCHITECTURE ─────────────────────────────────────────────

def build_neoncortrex_cnn(input_shape=(128, 128, 3), num_classes=4):
    """
    NeonCortrex-CNN: Original 9-stage architecture
    Designed specifically for brain MRI classification.
    No pre-trained weights — learned entirely from scratch.
    """
    inputs = layers.Input(shape=input_shape, name='mri_input')

    # Stem: initial feature extraction
    x = conv_block(inputs, 32, kernel_size=5)  # Large kernel for initial structure capture

    # Stage 1: 32 filters — detects low-level edges and textures
    x = neoncortrex_block(x, 32,  dropout_rate=0.1)   # 64→32

    # Stage 2: 64 filters — detects shapes and curves
    x = neoncortrex_block(x, 64,  dropout_rate=0.15)  # 32→16

    # Stage 3: 128 filters — detects complex tumor patterns
    x = neoncortrex_block(x, 128, dropout_rate=0.20)  # 16→8

    # Stage 4: 256 filters — high-level semantic features
    x = neoncortrex_block(x, 256, dropout_rate=0.25)  # 8→4

    # Global feature aggregation
    x = layers.GlobalAveragePooling2D(name='feature_vector')(x)

    # Classification head
    x = layers.Dense(256, activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='dense_256')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='dense_128')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax',
                            name='classification_output')(x)

    model = models.Model(inputs, outputs, name='NeonCortrex_CNN')
    return model


model = build_neoncortrex_cnn(input_shape=(*IMG_SIZE, 3), num_classes=4)
model.summary()
print(f"\nTotal parameters: {model.count_params():,}")

# ─── COMPILE ──────────────────────────────────────────────────────────────────

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LR,
    weight_decay=1e-5
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# ─── CALLBACKS ────────────────────────────────────────────────────────────────

callback_list = [
    # Save best model based on val_accuracy
    callbacks.ModelCheckpoint(
        MODEL_SAVE, monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
    # Reduce LR when validation accuracy plateaus
    callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5,
        patience=5, min_lr=1e-6, verbose=1
    ),
    # Stop early if no improvement for 12 epochs
    callbacks.EarlyStopping(
        monitor='val_accuracy', patience=12,
        restore_best_weights=True, verbose=1
    ),
]

# ─── TRAIN ────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  NeonCortrex-CNN Training Started")
print("  Custom architecture — no transfer learning")
print("="*60 + "\n")

history = model.fit(
    train_flow,
    epochs=EPOCHS,
    validation_data=val_flow,
    callbacks=callback_list,
    verbose=1
)

# ─── EVALUATE ─────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  Evaluating on Test Set")
print("="*60)
test_loss, test_acc, test_auc = model.evaluate(test_flow, verbose=1)
print(f"\nTest Accuracy : {test_acc*100:.2f}%")
print(f"Test AUC      : {test_auc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")
print(f"\nModel saved to: {MODEL_SAVE}")

# ─── PLOT TRAINING CURVES ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('NeonCortrex-CNN Accuracy'); axes[0].legend()
axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('NeonCortrex-CNN Loss'); axes[1].legend()
plt.tight_layout()
plt.savefig('../training_curves.png', dpi=150)
print("Training curves saved to training_curves.png")
plt.show()
