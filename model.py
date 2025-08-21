# ==============================================================================
# File: model.py
# Description: Defines the novel Sparse U-Net neural network architecture.
# ==============================================================================

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from config import INPUT_SHAPE, NUM_CLASSES, CARDINALITY

# --- Custom Model Blocks ---

def bnrelu(x):
    """A standard Batch Normalization -> ReLU activation block."""
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x

def conv_block(x, nb_filter, kernel_size=(3, 3), strides=(1, 1)):
    """A standard convolutional block."""
    x = Conv2D(nb_filter, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(x)
    x = bnrelu(x)
    return x

def add_common_layers(y):
    """Adds common layers for ResNeXt block."""
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y

def grouped_convolution(y, nb_channels, _strides):
    """Grouped convolution operation for ResNeXt."""
    if CARDINALITY == 1:
        return SeparableConv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    
    assert not nb_channels % CARDINALITY
    _d = nb_channels // CARDINALITY
    groups = []
    for j in range(CARDINALITY):
        group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
        groups.append(Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
    y = concatenate(groups)
    return y

def squeeze_excite_block(inputs, ratio=8):
    """Squeeze and Excite block for channel-wise feature recalibration."""
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = Multiply()([init, se])
    return x

def resnext_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1)):
    """ResNeXt block with squeeze-and-excite."""
    shortcut = y
    
    y = SeparableConv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    y = add_common_layers(y)
    
    y = grouped_convolution(y, nb_channels_in, _strides=_strides)
    y = add_common_layers(y)
    
    y = SeparableConv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    y = bnrelu(y)
    
    se = squeeze_excite_block(y)
    
    shortcut = SeparableConv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
    shortcut = bnrelu(shortcut)
    
    out = Add()([shortcut, se])
    out = Activation('relu')(out)
    return out

def exponential_index_fetch(ms_blocks_list):
    """Fetches outputs from previous blocks based on an exponential pattern."""
    num_blocks = len(ms_blocks_list)
    inputs = []
    i = 1
    while i <= num_blocks:
        inputs.append(ms_blocks_list[num_blocks - i])
        i = i * 2
    return inputs

def sparse_block(x, n_filters, num_blocks, growth_rate):
    """A block with sparse, long-range connections."""
    ms_blocks_list = []
    infil = 32
    outfil = 64
    
    for _ in range(num_blocks):
        sparse_out = resnext_block(x, infil, outfil)
        ms_blocks_list.append(sparse_out)
        
        fetch_outputs = exponential_index_fetch(ms_blocks_list)
        x = concatenate(fetch_outputs, axis=-1)
        
    return x

def upsample_concat_block(x, xskip, filters):
    """Upsampling block for the decoder path."""
    u = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
    out_se = squeeze_excite_block(xskip)
    out_con = concatenate([u, out_se])
    return out_con

# --- Model Assembly ---

def build_sparse_unet_model(input_shape, n_filters, num_blocks, growth_rate, dropout_rate, num_classes):
    """Builds the complete Sparse U-Net model."""
    inputs = Input(input_shape)
    
    # Encoder
    sp_block_0 = sparse_block(inputs, n_filters, num_blocks, growth_rate)
    p0 = MaxPooling2D(pool_size=(2, 2))(sp_block_0)
    
    sp_block_1 = sparse_block(p0, n_filters, num_blocks, growth_rate)
    p1 = MaxPooling2D(pool_size=(2, 2))(sp_block_1)
    
    # Bridge
    bridge = sparse_block(p1, n_filters, num_blocks, growth_rate)
    
    # Decoder
    u1 = upsample_concat_block(bridge, sp_block_1, n_filters)
    d1 = sparse_block(u1, n_filters, num_blocks, growth_rate)
    
    u2 = upsample_concat_block(d1, sp_block_0, n_filters)
    d2 = sparse_block(u2, n_filters, num_blocks, growth_rate)
    
    # Output
    out = bnrelu(d2)
    model_output = Dropout(dropout_rate)(out)
    model_output = Conv2D(num_classes, (1, 1), padding="same", activation='softmax')(model_output)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[model_output])
    return model
