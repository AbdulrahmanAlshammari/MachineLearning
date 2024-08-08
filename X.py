import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. Prepare Data Generators with Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)  

train_generator = datagen.flow_from_directory(
    'C:\\Users\\xxxx\\Documents\\dataset\\train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'C:\\Users\\xxxx\\Documents\\dataset\\validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'C:\\Users\\xxxx\\Documents\\dataset\\test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# 2. Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the Model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# 5. Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# 6. Save and Load the Model
model.save('tower_model.keras')  # Save in the newer Keras format

# Load the model
loaded_model = tf.keras.models.load_model('tower_model.keras')

# 7. Predict on New Data
def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = list(train_generator.class_indices.keys())  # Get class labels
    predicted_label = class_labels[predicted_class[0]]
    
    # Information about locations
    location_info = {
        'Location 1': 'Details of location 1',
        'Location 2': 'Details of location 2',
        'Location 3': 'details of lcoation 3'
    }
    
    print(f'Predicted Class: {predicted_label}')
    print(f'Information: {location_info.get(predicted_label, "No information available.")}')

# Example usage:
predict_image('xxxx')  # Replace with the path to the image you want to classify