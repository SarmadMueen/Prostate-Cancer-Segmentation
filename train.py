
# ==============================================================================
# File: train.py
# Description: The main script to run the training process.
# ==============================================================================


def main():
    """Main training function."""
    # Load data paths
    train_img_paths = sorted(glob.glob(os.path.join(TRAIN_DATA_DIR, '*.jpg')))
    train_mask_paths = sorted(glob.glob(os.path.join(TRAIN_MASK_DIR, '*.png')))
    valid_img_paths = sorted(glob.glob(os.path.join(VALID_DATA_DIR, '*.jpg')))
    valid_mask_paths = sorted(glob.glob(os.path.join(VALID_MASK_DIR, '*.png')))

    # Create data generators
    train_gen = DataGenerator(
        train_img_paths,
        train_mask_paths,
        batch_size=BATCH_SIZE,
        augmentation=get_training_augmentation(),
        shuffle=True
    )
    
    valid_gen = DataGenerator(
        valid_img_paths,
        valid_mask_paths,
        batch_size=BATCH_SIZE,
        augmentation=get_validation_augmentation(),
        shuffle=False
    )
    
    # Build the model
    model = build_attention_unet(INPUT_SHAPE, NUM_CLASSES)
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=iou_loss, metrics=['accuracy', mean_iou])
    
    model.summary()
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(MODEL_CHECKPOINT_PATH, save_best_only=True, monitor='val_mean_iou', mode='max'),
        EarlyStopping(patience=10, monitor='val_mean_iou', mode='max', restore_best_weights=True)
    ]
    
    # Train the model
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
