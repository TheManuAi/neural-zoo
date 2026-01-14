import os
import urllib.request
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# config
DATA_PATH = 'data'
MAX_SAMPLES_PER_CLASS = 15000
TEST_SIZE = 0.2
EPOCHS = 40
BATCH_SIZE = 128

# 50 animal categories
ANIMAL_CLASSES = [
    "ant", "bat", "bear", "bee", "bird", "butterfly", "camel", "cat", "cow", 
    "crab", "crocodile", "dog", "dolphin", "dragon", "duck", "elephant", "fish", 
    "flamingo", "frog", "giraffe", "hedgehog", "horse", "kangaroo", "lion", 
    "lobster", "mermaid", "monkey", "mosquito", "mouse", "octopus", "owl", 
    "panda", "parrot", "penguin", "pig", "rabbit", "raccoon", "rhinoceros", 
    "scorpion", "sea turtle", "shark", "sheep", "snail", "snake", "spider", 
    "squirrel", "swan", "tiger", "whale", "zebra"
]

def get_classes():
    classes = ANIMAL_CLASSES
    classes.sort()
    print(f"{len(classes)} classes.")
    return classes

def download_data(classes):
    os.makedirs(DATA_PATH, exist_ok=True)
    base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    
    print("Checking data...")
    for c in classes:
        safe_c = c.replace(' ', '%20')
        path = os.path.join(DATA_PATH, f'{c}.npy')
        if not os.path.exists(path):
            print(f'Downloading {c}...')
            try:
                urllib.request.urlretrieve(f'{base_url}{safe_c}.npy', path)
            except Exception as e:
                print(f"Error: {c}: {e}")
    print("Data ready.")

def load_and_preprocess_data(classes):
    available_classes = [c for c in classes if os.path.exists(os.path.join(DATA_PATH, f'{c}.npy'))]
    print(f"Loading {len(available_classes)} classes...")
    
    class_to_idx = {name: idx for idx, name in enumerate(available_classes)}
    X_data = []
    y_data = []

    for category in available_classes:
        try:
            data = np.load(os.path.join(DATA_PATH, f'{category}.npy'))
            data = data[:MAX_SAMPLES_PER_CLASS]
            data = data.astype('float32') / 255.0
            
            X_data.append(data)
            y_data.append(np.full(len(data), class_to_idx[category]))
        except Exception as e:
            print(f"Skipping {category}: {e}")

    if not X_data:
        raise ValueError("No data loaded.")

    X = np.concatenate(X_data, axis=0)
    y = np.concatenate(y_data, axis=0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, len(available_classes)

def build_model(num_classes, learning_rate=0.001, dropout_rate=0.4, dense_units=512, start_filters=64):
    inputs = layers.Input(shape=(28, 28, 1))
    
    # block 1
    x = layers.Conv2D(start_filters, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(start_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # block 2
    x = layers.Conv2D(start_filters * 2, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(start_filters * 2, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # block 3
    x = layers.Conv2D(start_filters * 4, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(start_filters * 4, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # block 4
    x = layers.Conv2D(start_filters * 8, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(dense_units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(dense_units // 2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

def main():
    classes = get_classes()
    download_data(classes)
    X_train, X_test, y_train, y_test, num_classes = load_and_preprocess_data(classes)
    
    # best hyperparameters from EDA experiments
    params = {
        'learning_rate': 0.001,
        'dropout_rate': 0.4,
        'dense_units': 512
    }
    
    print(f"Training with: {params}")
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max'
    )
    
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # light augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        fill_mode='nearest'
    )
    
    model = build_model(num_classes=num_classes, **params)
    print(f"Parameters: {model.count_params():,}")
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    
    best_acc = max(history.history['val_accuracy'])
    print(f"\nDone. Accuracy: {best_acc:.4f}")
    
    model.save('doodle_cnn_best.keras')
    print("Model saved to doodle_cnn_best.keras")

if __name__ == "__main__":
    main()
