from flask import Flask, render_template, request, send_from_directory, url_for
import os
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_PATH = 'model/blood_model.h5'
model = load_model(MODEL_PATH)

# Classes
CLASS_NAMES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Preprocess Function
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Home Route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict
            image_array = preprocess_image(filepath)
            prediction = model.predict(image_array)[0]
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = float(np.max(prediction)) * 100

            return render_template('index.html',
                       result=predicted_class,
                       confidence=round(confidence, 2),
                       image_file=filename,
                       class_names=CLASS_NAMES,
                       class_probs=[round(float(p) * 100, 2) for p in prediction])


        return render_template('index.html',
                       result="Error: No file selected",
                       confidence=None,
                       image_file=None,
                       class_names=[],
                       class_probs=[],
                       error=True)
    
    # Handle GET with defaults
    return render_template('index.html',
                       result=None,
                       confidence=None,
                       image_file=None,
                       class_names=[],
                       class_probs=[])


if __name__ == '__main__':  # Correct
    app.run(debug=True)
