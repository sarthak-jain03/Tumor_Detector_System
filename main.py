from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import gdown
from PIL import Image, ImageEnhance

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
# model = load_model('Model/model (1).h5')
model_path = "model.h5"
google_drive_file_ID = "15cd4yvLLuAyIZlnbdJPQz6uI4Pl3QkKD"

# If model file doesn't exist, download it from Google Drive
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={google_drive_file_ID}&confirm=t"
    gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# Class labels
class_labels = ['glioma','meningioma','notumor', 'pituitary']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type
def predict_tumor(image_path):
    image_size = 128
    img = load_img(image_path, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_class_index = np.argmax(predictions[0])
    confidence_score = np.max(predictions[0])

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score


# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            # Save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict the tumor
            result, confidence = predict_tumor(file_location)

            # Return result along with image path for display
            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/uploads/{file.filename}')

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
