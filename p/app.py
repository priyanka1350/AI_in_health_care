from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
model = load_model('pneumonia_detection_model.keras')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0  # Normalize to [0, 1] range
    return x

def model_predict(img_path, model):
    x = preprocess_image(img_path)
    preds = model.predict(x)
    return preds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        preds = model_predict(file_path, model)
        result = "Pneumonia Detected" if preds[0][0] > 0.5 else "No Pneumonia"
        confidence = float(preds[0][0] if preds[0][0] > 0.5 else 1 - preds[0][0])
        os.remove(file_path)  # Clean up the uploaded file
        return jsonify({'result': result, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
