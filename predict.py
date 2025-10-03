import tensorflow as tf
import numpy as np
import json
import cv2 # OpenCV for image processing

# Load the trained model
model = tf.keras.models.load_model('saved_model/plant_disease_model.h5')

# Load the class names
with open('saved_model/class_names.json', 'r') as f:
    class_names = json.load(f)

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) # Must be the same size as training images
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0) # Add a batch dimension
    return img

# --- Main Prediction Function ---
def predict_disease(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make a prediction
    predictions = model.predict(processed_image)
    
    # Get the index of the highest probability
    predicted_class_index = np.argmax(predictions[0])
    
    # Get the class name using the index
    # We need to use string keys for json
    predicted_class_name = class_names[str(predicted_class_index)]
    
    # Get the confidence score
    confidence = np.max(predictions[0])
    
    return predicted_class_name, confidence

#Example Use
if __name__ == '__main__':
    # You need an image to test. Find one from your dataset or the internet.
    # For example, let's pick an image from the validation set.
    # Make sure the path is correct for your system.
    test_image_path = 'Plant_leave_diseases_dataset_with_augmentation/Apple___Black_rot/015c00a8-b648-4453-9685-649033282a5c___JR_FrgE.S_2800.JPG'
    
    try:
        predicted_disease, confidence_score = predict_disease(test_image_path)
        print(f"Predicted Disease: {predicted_disease}")
        print(f"Confidence: {confidence_score:.2f}")
    except FileNotFoundError:
        print(f"Error: The file was not found at {test_image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")