from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import kagglehub
import logging
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(title="Kidney Classification API")


@app.get("/")
async def root():
    return {"status": "healthy", "message": "Kidney Classification API is running"}

# Global model variable
model = None

def load_model_from_kaggle():
    """Load model from Kaggle Hub with error handling"""
    global model
    try:
        MODEL_PATH = kagglehub.model_download(
            "aliamrali/kidney-classification/keras/v1"
        )
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        model = load_model(os.path.join(
            MODEL_PATH, 'resnet50_kidney_ct_augmented.h5'), compile=False)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Initialize model at startup
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    try:
        logger.info("Starting model initialization...")
        model = load_model_from_kaggle()
        logger.info("Model initialized successfully!")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


# Define class labels (modify according to your model's labels)
labels = ["Cyst", "Normal", "Stone", "Tumor"]

@app.post("/predict")
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
