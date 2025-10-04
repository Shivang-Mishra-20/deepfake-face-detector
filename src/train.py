import os
import json
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from preprocess import get_data_generators

# ✅ Correct dataset paths
TRAIN_DIR = "/content/data/real_vs_fake/real-vs-fake/train"
VAL_DIR   = "/content/data/real_vs_fake/real-vs-fake/valid"

BATCH_SIZE = 16
EPOCHS = 3   # start small

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def build_model(input_shape=(224, 224, 3)):
    try:
        print("⚡ Trying to load EfficientNetB0 with ImageNet weights...")
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    except Exception as e:
        print("⚠️ Could not load ImageNet weights. Falling back to random initialization.")
        print("Error was:", e)
        base = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)

    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("🔎 Checking dataset paths...")
    print("TRAIN_DIR:", TRAIN_DIR)
    print("VAL_DIR:", VAL_DIR)

    train_gen, val_gen = get_data_generators(TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE)
    print(f"✅ Loaded dataset: {train_gen.samples} train, {val_gen.samples} val")
    print(f"Classes: {train_gen.class_indices}")

    print("⚡ Building model...")
    model = build_model((IMG_SIZE[0], IMG_SIZE[1], 3))

    callbacks = [
        ModelCheckpoint(os.path.join(MODEL_DIR, 'deepfake_efficientnetb0.h5'),
                        save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy', mode='max'),
        ReduceLROnPlateau(patience=2, factor=0.5, monitor='val_loss')
    ]

    print("🚀 Starting training...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

    with open(os.path.join(MODEL_DIR, 'class_indices.json'), 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)

    print("🎉 Training complete. Best model saved to:", os.path.join(MODEL_DIR, 'deepfake_efficientnetb0.h5'))

if __name__ == '__main__':
    main()
