# Create a directory to save the model
os.makedirs('saved_model', exist_ok=True)
model.save('saved_model/plant_disease_model.h5')

# Save the class indices (mapping of class name to integer)
class_indices = train_generator.class_indices
# Invert the dictionary to map integers back to class names
class_names = {v: k for k, v in class_indices.items()}

with open('saved_model/class_names.json', 'w') as f:
    json.dump(class_names, f)

print("Model and class names saved successfully!")