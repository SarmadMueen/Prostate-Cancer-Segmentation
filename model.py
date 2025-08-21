
# ==============================================================================
# File: model.py
# Description: Defines the neural network architecture.
# ==============================================================================

import tensorflow as tf
from tensorflow.keras.layers import *
from config import INPUT_SHAPE, NUM_CLASSES
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import DataGenerator, get_training_augmentation, get_validation_augmentation
from model import build_attention_unet
from utils import iou_loss, mean_iou
from config import *


# --- Custom Model Blocks ---

def conv_block(x, num_filters, kernel_size=(3, 3), strides=(1, 1)):
    """A standard convolutional block with Batch Norm and ReLU activation."""
    x = Conv2D(num_filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attention_block(F_g, F_l, F_int):
    """Attention Gate (AG) for segmentation models."""
    g = Conv2D(F_int, (1, 1), strides=(1, 1), padding='valid')(F_g)
    g = BatchNormalization()(g)
    
    x = Conv2D(F_int, (1, 1), strides=(1, 1), padding='valid')(F_l)
    x = BatchNormalization()(x)
    
    psi = add([g, x])
    psi = Activation('relu')(psi)
    
    psi = Conv2D(1, (1, 1), strides=(1, 1), padding='valid')(psi)
    psi = Activation('sigmoid')(psi)
    
    return multiply([F_l, psi])

# --- Model Assembly ---

def build_attention_unet(input_shape, num_classes):
    """Builds a U-Net model with attention gates."""
    inputs = Input(input_shape)
    
    # Encoder
    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bridge
    c4 = conv_block(p3, 512)
    
    # Decoder
    u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    a5 = attention_block(u5, c3, 256)
    c5 = concatenate([u5, a5])
    c5 = conv_block(c5, 256)
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    a6 = attention_block(u6, c2, 128)
    c6 = concatenate([u6, a6])
    c6 = conv_block(c6, 128)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    a7 = attention_block(u7, c1, 64)
    c7 = concatenate([u7, a7])
    c7 = conv_block(c7, 64)
    
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c7)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model
