from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_dir, val_dir, img_size=(224,224), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb'   # ✅ force RGB images (3 channels)
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb'   # ✅ force RGB images (3 channels)
    )

    return train_gen, val_gen
