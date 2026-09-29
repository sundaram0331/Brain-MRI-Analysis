"""
NeonCortrex-SEG: Custom Encoder-Decoder Segmentation Model
Produces a pixel-level binary mask of the tumor region.
Built entirely from scratch — no U-Net weights used.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 16
EPOCHS      = 50
LR          = 1e-4
DATA_DIR    = '../dataset/br35h'      # Folder you will download
MODEL_SAVE  = '../neoncortrex_seg.keras'

# ─── CUSTOM BLOCKS ────────────────────────────────────────────────────────────

def encoder_block(x, filters, dropout=0.1):
    """Encoder: extract features + downsample."""
    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    skip = layers.ReLU()(x)                        # Save for skip connection
    x    = layers.MaxPooling2D(2)(skip)
    x    = layers.SpatialDropout2D(dropout)(x)
    return x, skip


def bottleneck(x, filters):
    """Deepest point — maximum feature compression."""
    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def decoder_block(x, skip, filters):
    """Decoder: upsample + merge skip connection."""
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip])             # Merge with encoder skip
    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_neoncortrex_seg(input_shape=(256, 256, 3)):
    """
    NeonCortrex-SEG: Custom Encoder-Decoder for tumor segmentation.
    Outputs a single-channel probability mask (sigmoid activation).
    """
    inputs = layers.Input(shape=input_shape, name='mri_seg_input')

    # Encoder (progressively extract features)
    x, s1 = encoder_block(inputs,  32, dropout=0.05)   # 256→128
    x, s2 = encoder_block(x,       64, dropout=0.10)   # 128→64
    x, s3 = encoder_block(x,      128, dropout=0.15)   # 64→32
    x, s4 = encoder_block(x,      256, dropout=0.20)   # 32→16

    # Bottleneck (16×16 — maximum compression)
    x = bottleneck(x, 512)

    # Decoder (progressively restore resolution)
    x = decoder_block(x, s4, 256)   # 16→32
    x = decoder_block(x, s3, 128)   # 32→64
    x = decoder_block(x, s2,  64)   # 64→128
    x = decoder_block(x, s1,  32)   # 128→256

    # Output mask: 1 channel, sigmoid = probability per pixel
    outputs = layers.Conv2D(1, 1, activation='sigmoid', name='seg_mask')(x)

    model = models.Model(inputs, outputs, name='NeonCortrex_SEG')
    return model


model = build_neoncortrex_seg()
model.summary()
print(f"Total parameters: {model.count_params():,}")
