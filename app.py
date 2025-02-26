import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained model
MODEL_PATH = 'models/diagnostic_model.h5'
model = None

def load_trained_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Model file not found. Please ensure the correct model path.")

load_trained_model()

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_disease(img_path):
    if model:
        img_array = preprocess_image(img_path)
        if img_array is not None:
            prediction = model.predict(img_array)
            result = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
            confidence = float(prediction[0][0]) * 100
            return result, confidence
    return 'Error processing image', 0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result, confidence = predict_disease(filepath)
            return render_template('result.html', result=result, confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
