
# ==============================================================================
# File: utils.py
# Description: Contains utility functions like losses, metrics, and callbacks.
# ==============================================================================

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

def iou_loss(y_true, y_pred):
    """Intersection over Union (IoU) loss function."""
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return 1.0 - (intersection + 1.0) / (union + 1.0)

def mean_iou(y_true, y_pred):
    """Mean IoU metric."""
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
    
    # Using Keras metric for simplicity and correctness
    m = tf.keras.metrics.MeanIoU(num_classes=tf.shape(y_pred)[-1])
    m.update_state(y_true, y_pred)
    return m.result()




if __name__ == '__main__':
    main()

