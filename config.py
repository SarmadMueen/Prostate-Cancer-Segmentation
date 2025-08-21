# ==============================================================================
# File: config.py
# Description: Central configuration file for paths, hyperparameters, etc.
# ==============================================================================

# --- PATHS ---
TRAIN_MASK_DIR = 'drive/My Drive/prostate/Gleason_train_masks/'
TRAIN_DATA_DIR = 'drive/My Drive/prostate/train_111_199_204/'
VALID_MASK_DIR = 'drive/My Drive/prostate/Gleason_valid_masks/'
VALID_DATA_DIR = 'drive/My Drive/prostate/valid_imgs_76/'
MODEL_CHECKPOINT_PATH = 'drive/My Drive/prostate/prostate_segmentation_model.h5'

# --- MODEL HYPERPARAMETERS ---
INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE = 5
EPOCHS = 100
LEARNING_RATE = 0.003
NUM_CLASSES = 5

# --- DATA AUGMENTATION & PREPROCESSING ---
# Colourmap for multi-class mask encoding
COLORMAP = {
    0: (0, 255, 0),      # Benign
    1: (0, 0, 255),      # Gleason 3
    2: (255, 255, 0),    # Gleason 4
    3: (255, 0, 0),      # Gleason 5
    4: (255, 255, 255)   # Ignore/White
}
