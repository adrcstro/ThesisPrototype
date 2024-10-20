# Step 1: Import Required Libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.model_selection import train_test_split



#Step 2: Load Your LBP Processed Dataset

# Load your LBP-processed dataset
X = np.load('/path/to/your/lbp_images.npy')  # Your LBP processed images
y = np.load('/path/to/your/labels.npy')  # Corresponding labels for the images

# Ensure the dataset is reshaped correctly for grayscale input (e.g., (128, 128, 1))
X = np.expand_dims(X, axis=-1)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 3: Build the MobileNetV2 Model
def build_mobilenet_model(input_shape):
    base_model = applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None)

    # Add a custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')  # Adjust for the number of classes in your dataset
    ])
    
    return model

input_shape = (128, 128, 1)  # Grayscale images, adjust this according to your image size
model = build_mobilenet_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#Step 4: Train the Model
# Augment training data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# Train the model
history = model.fit(train_generator, validation_data=(X_test, y_test), epochs=50)

#Step 5: Evaluate the Model
# Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

#Step 6: Save the Model
# Save model after training
model.save('mobilenet_v2_skin_disease.h5')

















