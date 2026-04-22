import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from dataset_loader import load_bird_dataset

def build_elitedensenet(num_classes):
    """
    Builds the EliteDenseNet model based on DenseNet121 backbone.
    Returns both the composite model and the backbone for two-phase unfreezing.
    """
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False # Initial freeze for Phase 1

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model, base_model

def train_model():
    print("Preparing to train EliteDenseNet with 90%+ Accuracy Strategy...")
    
    train_ds, val_ds, class_names = load_bird_dataset(batch_size=32, debug_fast_run=False)
    num_classes = len(class_names)
    
    model, base_model = build_elitedensenet(num_classes)
    
    checkpoint_filepath = '../data/elitedensenet_model.h5'
    os.makedirs('../data', exist_ok=True)

    # Elite Callbacks
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, save_weights_only=False,
        monitor='val_accuracy', mode='max', save_best_only=True)
        
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7)

    # PHASE 1: Warmup
    print("Starting Phase 1: Feature Head Warmup (15 Epochs)...")
    model.fit(
        train_ds, validation_data=val_ds, epochs=15,
        callbacks=[model_checkpoint, early_stop, reduce_lr]
    )

    # PHASE 2: Deep Fine-Tuning
    print("Starting Phase 2: Deep Fine-Tuning...")
    base_model.trainable = True
    
    # Freeze the bottom 150 layers to retain fundamental ImageNet structures
    for layer in base_model.layers[:150]:
        layer.trainable = False

    # Recompile with ultra-low learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.fit(
        train_ds, validation_data=val_ds, epochs=30,
        callbacks=[model_checkpoint, early_stop, reduce_lr]
    )

    print(f"Two-phase explicit fine-tuning completed. Ultimate Model saved to {checkpoint_filepath}.")

    with open('../data/class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

if __name__ == "__main__":
    train_model()
