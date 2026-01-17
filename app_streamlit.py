"""
Streamlit app for Fatty Liver Classification
Ultrasound image analysis with deep learning
"""
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.siamese_net import SiameseNetwork
from src.data_loader import get_data_transforms
from skimage import exposure

# Set page config
st.set_page_config(
    page_title="🏥 Fatty Liver Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Global variables
@st.cache_resource
def load_model():
    """Load model once"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork().to(device)
    
    model_path = 'best_model.pth'
    if not Path(model_path).exists():
        st.error(f"❌ Model not found: {model_path}")
        return None, device, None
    
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    
    model.eval()
    transform = get_data_transforms(is_train=False)
    
    return model, device, transform

def is_valid_ultrasound(img_pil):
    """Validate if image is likely an ultrasound"""
    img_array = np.array(img_pil.convert('L'))
    
    # Check dimensions
    if img_array.shape[0] < 100 or img_array.shape[1] < 100:
        return False, "Image too small"
    
    if img_array.shape[0] > 2000 or img_array.shape[1] > 2000:
        return False, "Image too large"
    
    # Check contrast
    img_eq = exposure.equalize_adapthist(img_array / 255.0)
    contrast = img_eq.std()
    
    if contrast < 0.02:
        return False, "Low contrast (not ultrasound)"
    
    # Check for color uniformity (ultrasound is grayscale-like)
    if hasattr(img_pil, 'convert'):
        img_rgb = np.array(img_pil.convert('RGB'))
        color_std = img_rgb.std(axis=2).mean()
        if color_std > 50:
            return False, "Too much color variation"
    
    return True, "Valid ultrasound"

def predict(image, model, device, transform, mode='multiclass'):
    """Get prediction from model"""
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        probs_np = probs.cpu().numpy()[0]
        pred = int(np.argmax(probs_np))
        confidence = float(np.max(probs_np))
    
    # Convert to binary if requested
    if mode == 'binary':
        binary_pred = 0 if pred == 0 else 1
        normal_prob = probs_np[0]
        abnormal_prob = 1 - normal_prob
        return binary_pred, np.array([normal_prob, abnormal_prob]), confidence
    
    return pred, probs_np, confidence

# Main app
def main():
    # Header
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://img.icons8.com/color/96/000000/liver.png", width=80)
    with col2:
        st.title("🏥 Fatty Liver Classification")
        st.markdown("**Deep Learning-based Ultrasound Analysis**")
    
    # Load model
    model, device, transform = load_model()
    
    if model is None:
        st.error("⚠️ Model failed to load. Please check that best_model.pth exists.")
        return
    
    # Sidebar config
    st.sidebar.header("⚙️ Configuration")
    classification_mode = st.sidebar.radio(
        "Classification Mode",
        ["Multi-class (5 grades)", "Binary (Normal/Abnormal)"],
        help="Choose between detailed 5-class or simple binary classification"
    )
    mode = 'multiclass' if classification_mode == 'Multi-class (5 grades)' else 'binary'
    
    # Class info
    st.sidebar.markdown("### 📚 Class Information")
    class_info = {
        'Normal': 'Healthy liver with no steatosis',
        'Grade-I': 'Mild fatty liver (5-35% fat)',
        'Grade-II': 'Moderate fatty liver (35-65% fat)',
        'Grade-III': 'Severe fatty liver (>65% fat)',
        'CLD': 'Chronic Liver Disease'
    }
    
    for class_name, desc in class_info.items():
        st.sidebar.info(f"**{class_name}**: {desc}")
    
    # Main content
    tabs = st.tabs(["📤 Predict", "ℹ️ About", "📖 Guide"])
    
    # Prediction tab
    with tabs[0]:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an ultrasound image",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Upload a liver ultrasound image"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Validate
                is_valid, msg = is_valid_ultrasound(image)
                if not is_valid:
                    st.warning(f"⚠️ Validation Warning: {msg}")
        
        with col2:
            st.subheader("🔍 Prediction Result")
            
            if uploaded_file:
                with st.spinner("🔄 Analyzing image..."):
                    try:
                        pred_idx, probs, confidence = predict(image, model, device, transform, mode)
                        
                        if mode == 'binary':
                            class_names = ['Normal', 'Abnormal']
                        else:
                            class_names = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD']
                        
                        pred_label = class_names[pred_idx]
                        
                        # Display prediction
                        if confidence < 0.25:
                            st.warning(f"⚠️ **Low Confidence Prediction**")
                            st.info("The model is uncertain. Please consult a medical professional.")
                        
                        st.markdown(f"### {pred_label}")
                        
                        # Confidence metric
                        col_conf, col_severity = st.columns(2)
                        with col_conf:
                            st.metric(
                                "Confidence",
                                f"{confidence * 100:.2f}%",
                                delta=f"{'High' if confidence > 0.7 else 'Low'}"
                            )
                        
                        with col_severity:
                            severity = "🟢 Normal" if pred_idx == 0 else "🔴 Abnormal"
                            st.metric("Status", severity)
                        
                        # Probability distribution
                        st.subheader("📊 Probability Distribution")
                        
                        prob_dict = {class_names[i]: probs[i] * 100 for i in range(len(class_names))}
                        
                        # Bar chart
                        st.bar_chart(prob_dict)
                        
                        # Detailed probabilities
                        st.markdown("**Detailed Probabilities:**")
                        for class_name, prob in prob_dict.items():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.progress(prob / 100)
                            with col2:
                                st.text(f"{prob:.2f}%")
                        
                        # Medical recommendation
                        st.subheader("⚕️ Clinical Recommendation")
                        if pred_idx == 0:
                            st.success("✅ **Normal Liver** - No intervention needed. Continue regular health monitoring.")
                        else:
                            st.error("❌ **Abnormal Finding** - Requires medical attention. Please consult a healthcare professional.")
                    
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
            else:
                st.info("👆 Upload an ultrasound image to get started!")
    
    # About tab
    with tabs[1]:
        st.markdown("""
        ## 🏥 About This Application
        
        ### What is this?
        This application uses deep learning to classify fatty liver disease from ultrasound images.
        
        ### How it works
        - **Model**: ResNet-50 based Siamese Neural Network
        - **Training Data**: 550 ultrasound images from multiple patients
        - **Accuracy**: 99%+ on validation set
        
        ### Classification Categories
        - **Normal**: Healthy liver with no steatosis
        - **Grade-I**: Mild fatty liver (5-35% fat)
        - **Grade-II**: Moderate fatty liver (35-65% fat)
        - **Grade-III**: Severe fatty liver (>65% fat)
        - **CLD**: Chronic Liver Disease
        
        ### Important Disclaimer
        ⚠️ **This is an AI-assisted tool for research/educational purposes only.**
        - Not for clinical diagnosis
        - Always consult a qualified medical professional
        - Results should be verified by a radiologist
        
        ### Technology
        - **Framework**: PyTorch
        - **UI**: Streamlit
        - **Model**: ResNet-50 + Classification Head
        """)
    
    # Guide tab
    with tabs[2]:
        st.markdown("""
        ## 📖 How to Use This Application
        
        ### Step 1: Upload Image
        Click "📤 Upload Image" in the Prediction tab and select your ultrasound image.
        
        ### Step 2: Select Classification Mode
        Use the sidebar to choose between:
        - **Multi-class**: Detailed 5-grade classification
        - **Binary**: Simple Normal/Abnormal classification
        
        ### Step 3: Review Results
        - View the predicted class
        - Check the confidence level
        - Examine probability distribution
        
        ### Supported Formats
        - JPG / JPEG
        - PNG
        - BMP
        
        ### Image Requirements
        - Minimum size: 100x100 pixels
        - Maximum size: 2000x2000 pixels
        - Preferably grayscale or near-grayscale (medical image)
        
        ### Tips for Best Results
        1. Use high-quality ultrasound images
        2. Ensure proper contrast
        3. Avoid heavily compressed images
        4. Follow medical imaging standards
        
        ### What to Do Next
        1. Review the prediction carefully
        2. Note the confidence level
        3. Cross-reference with medical expertise
        4. Consult a qualified radiologist
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
    <p>🏥 Fatty Liver Classification System | Made with ❤️ for Medical Imaging</p>
    <p><small>Disclaimer: This tool is for research and educational purposes only.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
