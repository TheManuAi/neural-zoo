import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = 'data'
MAX_SAMPLES_PER_CLASS = 15000
TEST_SIZE = 0.2
BATCH_SIZE = 128
EPOCHS = 50

# Target classes
ANIMAL_CLASSES = [
    "ant", "bat", "bear", "bee", "bird", "butterfly", "camel", "cat", "cow", 
    "crab", "crocodile", "dog", "dolphin", "dragon", "duck", "elephant", "fish", 
    "flamingo", "frog", "giraffe", "hedgehog", "horse", "kangaroo", "lion", 
    "lobster", "mermaid", "monkey", "mosquito", "mouse", "octopus", "owl", 
    "panda", "parrot", "penguin", "pig", "rabbit", "raccoon", "rhinoceros", 
    "scorpion", "sea turtle", "shark", "sheep", "snail", "snake", "spider", 
    "squirrel", "swan", "tiger", "whale", "zebra"
]

def load_data():
    classes = sorted([c for c in ANIMAL_CLASSES if os.path.exists(os.path.join(DATA_PATH, f'{c}.npy'))])
    
    print(f"Loading {len(classes)} classes with {MAX_SAMPLES_PER_CLASS} samples each...")
    
    X_data, y_data = [], []
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    
    for c in classes:
        data = np.load(os.path.join(DATA_PATH, f'{c}.npy'))
        data = data[:MAX_SAMPLES_PER_CLASS].astype('float32') / 255.0
        X_data.append(data)
        y_data.append(np.full(len(data), class_to_idx[c]))
    
    X = np.concatenate(X_data)
    y = np.concatenate(y_data)
    
    print(f"Total samples: {len(X)}")
    return X, y, classes

def build_deep_cnn(num_classes):
    inputs = layers.Input(shape=(28, 28, 1))
    
    # Block 1
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 4
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Global pooling is more robust than flattening
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classifier head
    x = layers.Dense(1024, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

def cosine_decay_with_warmup(epoch, total_epochs=EPOCHS, warmup_epochs=5, initial_lr=0.001, min_lr=1e-6):
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))

def train_model():
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    X, y, classes = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    # Reshape for CNN input
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    print("\nBuilding model...")
    model = build_deep_cnn(len(classes))
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    print(f"Parameters: {model.count_params():,}")
    
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        callbacks.LearningRateScheduler(
            lambda epoch: cosine_decay_with_warmup(epoch),
            verbose=0
        )
    ]
    
    # Data augmentation settings
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    print("\nTraining...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list,
        verbose=1
    )
    
    best_val_acc = max(history.history['val_accuracy'])
    final_val_acc = history.history['val_accuracy'][-1]
    
    print("\n" + "=" * 70)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print("=" * 70)
    
    model.save('doodle_cnn_best.keras')
    print("\nModel saved to 'doodle_cnn_best.keras'")
    
    with open('classes.txt', 'w') as f:
        f.write('\n'.join(classes))
    print("Classes saved to 'classes.txt'")
    
    plot_history(history)
    return model, history

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    print("Plot saved to 'training_history.png'")

if __name__ == "__main__":
    train_model()
