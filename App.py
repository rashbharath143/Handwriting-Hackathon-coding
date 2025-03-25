import os
import logging
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from model import load_model, preprocess_image, predict_digit
from utils import allowed_file


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
# Load the ML model
logger.info("Initializing digit recognition model")
model = load_model()
logger.info("Model initialized successfully")
@app.route('/')
def index():
    """Render the main page of the application."""
    logger.debug("Serving index page")
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to process and predict digits from uploaded images."""
    try:
        logger.debug("Received request to /predict endpoint")
        
        # Check if image file was uploaded
        if 'image' not in request.files:
            logger.warning("No image uploaded")
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        # Check if the file is empty
        if file.filename == '':
            logger.warning("Empty file selected")
            return jsonify({'error': 'No image selected'}), 400
        
        # Check if the file format is allowed
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file format: {file.filename}")
            return jsonify({'error': 'File format not supported. Please upload JPG or PNG images.'}), 400
        
        # Read the image file
        logger.debug(f"Processing image: {file.filename}")
        image = Image.open(file)
        
        # Preprocess the image for the model
-0
+1
        
        # Make prediction
        digit, confidence = predict_digit(model, processed_image)
        logger.info(f"Prediction result: digit={digit}, confidence={confidence:.2f}")
        
        # Return the prediction result
        return jsonify({
-1
+4
        })
        
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500
@app.route('/predict_from_data', methods=['POST'])
def predict_from_data():
    """Endpoint to process and predict digits from base64 image data."""
    try:
        logger.debug("Received request to /predict_from_data endpoint")
        
        data = request.json
        if not data or 'image_data' not in data:
            logger.warning("No image data provided")
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get the base64 encoded image data
-0
+1
            image_data = image_data.split(',')[1]
        
        # Decode the base64 image
        logger.debug("Decoding base64 image data")
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
-0
+1
        
        # Make prediction
        digit, confidence = predict_digit(model, processed_image)
        logger.info(f"Prediction result: digit={digit}, confidence={confidence:.2f}")
        
        # Return the prediction result
        return jsonify({
-1
+1
        })
        
    except Exception as e:
        app.logger.error(f"Error during prediction from data: {str(e)}")
        logger.error(f"Error during prediction from data: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing image data: {str(e)}'}), 500
if __name__ == '__main__':
