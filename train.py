import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


DATASET_PATH = 'PlantVillage-Dataset/raw/color'

# The rest of the constants remain the same.
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10



# Create an ImageDataGenerator with data augmentation for the training set
# and rescaling for the validation set.
datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to [0, 1]
    rotation_range=40,           # Randomly rotate images
    width_shift_range=0.2,       # Randomly shift images horizontally
    height_shift_range=0.2,      # Randomly shift images vertically
    shear_range=0.2,             # Shear transformation
    zoom_range=0.2,              # Randomly zoom in on images
    horizontal_flip=True,        # Randomly flip images horizontally
    fill_mode='nearest',         # Strategy for filling in newly created pixels
    validation_split=0.2         # Reserve 20% of data for validation
)

# Create the training data generator
print("Loading Training Data...")
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training' # Set as training data
)

# Create the validation data generator
print("Loading Validation Data...")
validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' # Set as validation data
)


# Get the number of classes automatically from the generator
NUM_CLASSES = train_generator.num_classes
print(f"\nFound {NUM_CLASSES} classes.")

# Load the base model (MobileNetV2) without the top classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the convolutional base
base_model.trainable = False

# Add our custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)


# --- 3.5. Compile the Model ---
print("\nCompiling Model...")
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 3.6. Train the Model ---
print("Starting Model Training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE
)
print("Model Training Finished.")


# --- 3.7. Evaluate the Model (Visualize Training) ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



# Create a directory to save the model
os.makedirs('saved_model', exist_ok=True)
model.save('saved_model/plant_disease_model.keras')

# Save the class indices (mapping of class name to integer)
class_indices = train_generator.class_indices
# Invert the dictionary to map integers back to class names
class_names = {v: k for k, v in class_indices.items()}

with open('saved_model/class_names.json', 'w') as f:
    json.dump(class_names, f)

print("Model and class names saved successfully!")