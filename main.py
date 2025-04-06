from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Initialize FastAPI app
app = FastAPI()

# Load Model once
model = tf.keras.models.load_model('resnet50_kidney_ct_augmented.h5')


# Define class labels (modify according to your model's labels)
labels = ["Cyst", "Normal", "Stone", "Tumor"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image from the request
    img = Image.open(io.BytesIO(await file.read()))
    img = img.convert('RGB')  # Convert image to RGB if it's in another format

    # Resize and preprocess image to fit ResNet50 input format
    img = img.resize((224, 224))  # ResNet50 expects 224x224 images
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for ResNet50

    # Model prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)  # Get the class index
    
    # Return the prediction label
    return {"prediction": labels[predicted_class[0]]}