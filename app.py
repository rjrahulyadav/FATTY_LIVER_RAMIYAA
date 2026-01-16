"""
Flask web app for Fatty Liver Classification
Upload ultrasound images and get predictions
"""
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import time

from models.siamese_net import SiameseNetwork
from src.data_loader import get_data_transforms

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

CLASS_NAMES = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD']
CLASS_DESCRIPTIONS = {
    'Normal': 'Healthy liver with no steatosis',
    'Grade-I': 'Mild fatty liver (5-35% fat)',
    'Grade-II': 'Moderate fatty liver (35-65% fat)',
    'Grade-III': 'Severe fatty liver (>65% fat)',
    'CLD': 'Chronic Liver Disease'
}

# Global model and device
model = None
device = None
transform = None


def init_model():
    """Initialize model on first run"""
    global model, device, transform
    if model is not None:
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_data_transforms(is_train=False)
    
    model = SiameseNetwork().to(device)
    model_path = 'best_model.pth'
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f'Model not found: {model_path}. Please train the model first.')
    
    state = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except:
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'], strict=False)
    
    model.eval()
    print(f"Model loaded on {device}")


@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict fatty liver class from uploaded image"""
    try:
        init_model()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp'}
        if not any(file.filename.lower().endswith('.' + ext) for ext in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or BMP'}), 400
        
        start_time = time.time()
        
        # Load image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Preprocess
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            probs_np = probs.cpu().numpy()[0]
            pred = int(np.argmax(probs_np))
            confidence = float(np.max(probs_np))
        
        processing_time = time.time() - start_time
        
        # Prepare response
        result = {
            'status': 'success',
            'prediction': CLASS_NAMES[pred],
            'confidence': round(confidence * 100, 2),
            'description': CLASS_DESCRIPTIONS[CLASS_NAMES[pred]],
            'probabilities': {
                CLASS_NAMES[i]: round(float(probs_np[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            'processing_time': round(processing_time, 3),
            'severity': 'Abnormal - Requires Medical Attention' if pred != 0 else 'Normal',
            'device': str(device)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """About page"""
    return jsonify({
        'model': 'Siamese Neural Network',
        'classes': CLASS_NAMES,
        'descriptions': CLASS_DESCRIPTIONS,
        'device': str(device)
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
