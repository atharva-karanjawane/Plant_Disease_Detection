from flask import Blueprint, render_template, request, jsonify, current_app
import os
import numpy as np
from PIL import Image
import io
import base64
from app.utils import preprocess_image, get_prediction, generate_gradcam
from models.efficientnet import CropDiseaseModel

main = Blueprint('main', __name__)

# Initialize model
model = None

@main.before_app_first_request
def load_model():
    global model
    model = CropDiseaseModel()
    model.load_weights(current_app.config['MODEL_PATH'])

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    img = Image.open(io.BytesIO(file.read()))
    processed_img = preprocess_image(img)
    
    prediction, confidence = get_prediction(model, processed_img)
    class_name = current_app.config['CLASS_NAMES'][prediction]
    
    gradcam_img = generate_gradcam(model, processed_img, prediction)
    
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    buffered = io.BytesIO()
    gradcam_img.save(buffered, format="JPEG")
    gradcam_str = base64.b64encode(buffered.getvalue()).decode()
    
    display_name = class_name.replace('___', ' - ').replace('_', ' ')
    
    is_healthy = 'healthy' in class_name
    status = 'Healthy' if is_healthy else 'Diseased'
    
    recommendations = []
    if not is_healthy:
        if 'Apple_scab' in class_name:
            recommendations = [
                "Apply fungicides like captan or sulfur",
                "Prune infected branches during dormant season",
                "Rake and destroy fallen leaves"
            ]
        elif 'Black_rot' in class_name:
            recommendations = [
                "Remove mummified fruits from trees",
                "Apply fungicides during growing season",
                "Ensure good air circulation by proper pruning"
            ]
        elif 'blight' in class_name.lower():
            recommendations = [
                "Apply copper-based fungicides",
                "Avoid overhead irrigation",
                "Rotate crops in the affected area"
            ]
        elif 'rust' in class_name.lower():
            recommendations = [
                "Remove nearby cedar trees (for apple rust)",
                "Apply fungicides preventatively",
                "Ensure good air circulation"
            ]
        elif 'spot' in class_name.lower():
            recommendations = [
                "Apply appropriate fungicides",
                "Avoid wetting leaves during irrigation",
                "Remove and destroy infected plant parts"
            ]
        elif 'mosaic' in class_name.lower() or 'virus' in class_name.lower():
            recommendations = [
                "Remove and destroy infected plants",
                "Control insect vectors like aphids",
                "Use virus-resistant varieties for future planting"
            ]
        else:
            recommendations = [
                "Apply appropriate fungicides or bactericides",
                "Improve air circulation around plants",
                "Remove and destroy infected plant parts"
            ]
    else:
        recommendations = [
            "Continue regular monitoring",
            "Maintain good watering practices",
            "Apply balanced fertilization"
        ]
    
    return render_template('result.html', 
                          original_image=img_str,
                          gradcam_image=gradcam_str,
                          class_name=display_name,
                          confidence=float(confidence) * 100,
                          status=status,
                          recommendations=recommendations)

@main.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
 
    img = Image.open(io.BytesIO(file.read()))
    processed_img = preprocess_image(img)

    prediction, confidence = get_prediction(model, processed_img)
    class_name = current_app.config['CLASS_NAMES'][prediction]
    
    display_name = class_name.replace('___', ' - ').replace('_', ' ')
    
    return jsonify({
        'prediction': display_name,
        'confidence': float(confidence) * 100,
        'class_index': int(prediction)
    })