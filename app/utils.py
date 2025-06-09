import numpy as np
from PIL import Image
import tensorflow as tf
from models.gradcam import GradCAM
import cv2

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess an image for model prediction."""
 
    if image.size != target_size:
        image = image.resize(target_size)
    
    img_array = np.array(image) / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_prediction(model, preprocessed_image):
    """Get prediction from model."""
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return predicted_class, confidence

def generate_gradcam(model, preprocessed_image, predicted_class):
    """Generate Grad-CAM visualization."""
  
    gradcam = GradCAM(model)
    
    heatmap = gradcam.compute_heatmap(preprocessed_image, predicted_class)
    
    img = (preprocessed_image[0] * 255).astype(np.uint8)
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    superimposed_img = Image.fromarray(superimposed_img)
    
    return superimposed_img