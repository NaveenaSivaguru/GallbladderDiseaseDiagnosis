import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import pickle
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pathlib
from pathlib import Path
import base64

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DomainAdversarialLayer(nn.Module):
    def __init__(self, input_features, hidden_size=256):
        super(DomainAdversarialLayer, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x, current_epoch, total_epochs):
        alpha = 2.0 * (current_epoch / total_epochs) - 1.0
        alpha = np.exp(-10 * alpha)
        alpha = 2 / (1 + alpha) - 1
        reverse_x = GradientReversal.apply(x, alpha)
        domain_output = self.domain_classifier(reverse_x)
        return domain_output

class MorphologicalLayer(nn.Module):
    def __init__(self, operation="dilation", kernel_size=3):
        super(MorphologicalLayer, self).__init__()
        self.operation = operation
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=self.padding)

    def forward(self, x):
        if self.operation == "erosion":
            return -self.pool(-x)
        return self.pool(x)

class DualBranchDeepGCNN(nn.Module):
    def __init__(self, num_classes):
        super(DualBranchDeepGCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.in3 = nn.InstanceNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.in4 = nn.InstanceNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.morph_layer1 = MorphologicalLayer(operation="dilation", kernel_size=3)
        self.morph_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.in_morph1 = nn.InstanceNorm2d(32)
        self.morph_layer2 = MorphologicalLayer(operation="erosion", kernel_size=3)
        self.morph_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.in_morph2 = nn.InstanceNorm2d(64)
        self.morph_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.in_morph3 = nn.InstanceNorm2d(128)
        self.morph_conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.in_morph4 = nn.InstanceNorm2d(256)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(256 * 7 * 7 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.domain_adversarial_final = DomainAdversarialLayer(256 * 7 * 7 * 2)
        self.domain_adversarial_cnn = DomainAdversarialLayer(256 * 7 * 7)
        self.domain_adversarial_morph = DomainAdversarialLayer(256 * 7 * 7)
        self.gradients = None
        self.feature_maps = None

    def save_gradient(self, grad):
        self.gradients = grad

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def forward(self, x, current_epoch=0, total_epochs=1):
        x1 = self.pool(F.relu(self.in1(self.conv1(x))))
        x1 = self.pool(F.relu(self.in2(self.conv2(x1))))
        x1 = self.pool(F.relu(self.in3(self.conv3(x1))))
        x1 = self.pool(F.relu(self.in4(self.conv4(x1))))
        x1 = self.adaptive_pool(x1)
        x1_flat = torch.flatten(x1, start_dim=1)
        domain_output_cnn = self.domain_adversarial_cnn(x1_flat, current_epoch, total_epochs)
        x2 = self.morph_layer1(x)
        x2 = self.pool(F.relu(self.in_morph1(self.morph_conv1(x))))
        x2 = self.morph_layer2(x2)
        x2 = self.pool(F.relu(self.in_morph2(self.morph_conv2(x2))))
        x2 = self.pool(F.relu(self.in_morph3(self.morph_conv3(x2))))
        x2 = self.pool(F.relu(self.in_morph4(self.morph_conv4(x2))))
        x2 = self.adaptive_pool(x2)
        x2_flat = torch.flatten(x2, start_dim=1)
        domain_output_morph = self.domain_adversarial_morph(x2_flat, current_epoch, total_epochs)
        x = torch.cat((x1, x2), dim=1)
        x.requires_grad_(True)
        x.register_hook(self.save_gradient)
        self.save_feature_maps(None, None, x)
        x_flattened = torch.flatten(x, start_dim=1)
        domain_output_final = self.domain_adversarial_final(x_flattened, current_epoch, total_epochs)
        x = self.dropout(F.relu(self.fc1(x_flattened)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x, domain_output_final, domain_output_cnn, domain_output_morph

def load_model(model_path, num_classes, device):
    model = DualBranchDeepGCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def load_encoder(encoder_path):
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    return encoder

def predict_image(model, image_tensor, encoder, device):
    with torch.no_grad():
        task_output, _, _, _ = model(image_tensor)
        probabilities = torch.nn.functional.softmax(task_output, dim=1)
        _, predicted = torch.max(task_output, 1)
        predicted_class_index = predicted.item()
        predicted_class_label = encoder.inverse_transform([predicted_class_index])[0]
        return predicted_class_label, probabilities, predicted_class_index

def preprocess_numpy_image(image, mean, std, device):
    IMAGE_SIZE = (224, 224)
    image = cv2.resize(image, IMAGE_SIZE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 2:
        clahe_applied = clahe.apply(image)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        l_channel_clahe = clahe.apply(l_channel)
        lab_clahe = cv2.merge((l_channel_clahe, a, b))
        clahe_applied = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    image_pil = Image.fromarray(np.uint8(clahe_applied))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    return image_tensor

def main():
    # Custom CSS for the entire app
    icon_path = os.path.join(os.getcwd(),"ICON.png")
    st.set_page_config(
        page_title="Gallbladder Disease Diagnosis",
        page_icon=icon_path if os.path.exists(icon_path) else None,
        layout="centered"
    )
    st.markdown("""
        <style>
        :root {
            --primary: #2E86C1;
            --secondary: #F4D03F;
            --accent: #E74C3C;
            --light: #F8F9F9;
            --dark: #1B2631;
            --success: #28B463;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        }
        
        .title-container {
            background: linear-gradient(90deg, var(--primary) 0%, #3498db 100%);
            padding: 2rem;
            border-radius: 0 0 20px 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        
        .title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .upload-box {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            margin-bottom: 2rem;
        }
        
        .camera-box {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            margin-bottom: 2rem;
        }
        
        .result-box {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            margin-top: 1rem;
            border-left: 5px solid var(--primary);
            transition: all 0.3s ease;
        }
        
        .result-box:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        }
        
        .probability-bar {
            height: 25px;
            border-radius: 12px;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #e0e0e0 0%, #e0e0e0 100%);
            position: relative;
            overflow: hidden;
        }
        
        .probability-fill {
            height: 100%;
            border-radius: 12px;
            background: linear-gradient(90deg, var(--primary) 0%, #5dade2 100%);
            transition: width 0.5s ease;
        }
        
        .probability-label {
            position: absolute;
            width: 100%;
            text-align: center;
            line-height: 25px;
            color: white;
            font-weight: bold;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }
        
        .tab-container {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background: linear-gradient(90deg, var(--primary) 0%, #3498db 100%);
            padding: 0;
        }
               
        /* NEW: Increase ONLY tab text size */
        .stTabs [data-baseweb="tab"] span {
            font-size: 80px !important;
            line-height: 3.5 !important;
        }
                          
        /* Increase tab size */
        .stTabs [data-baseweb="tab"] {
            height: 70px;  /* Increased height */
            padding: 0 2rem !important;  /* Increased horizontal padding */
            font-size: 3.0rem !important;  /* Larger font size */
            font-weight: 600 !important;
        }
    
        /* Adjust the tab list container to accommodate larger tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;  /* Space between tabs */
        }
    
        /* Optional: Make active tab more prominent */
        .stTabs [aria-selected="true"] {
            background-color: rgba(255,255,255,0.3) !important;
            font-weight: 700 !important;
        }
        
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding: 1rem;
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        
        .camera-btn {
            background: linear-gradient(90deg, var(--accent) 0%, #e67e22 100%) !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 30px !important;
            padding: 0.5rem 1.5rem !important;
        }
        
        .upload-btn {
            background: linear-gradient(90deg, var(--success) 0%, #2ecc71 100%) !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 30px !important;
            padding: 0.5rem 1.5rem !important;
        }
        
        .disease-icon {
            font-size: 2.5rem;
            margin-right: 10px;
            color: var(--primary);
        }
        
        /* Added style for the app logo */
        .app-logo img {
            max-width: 350px;
            height: auto;
            display: block;
            margin: 0 auto 1rem auto;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display the logo at the top
    if os.path.exists(icon_path):
        with open(icon_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()

        st.markdown(
            f"""
            <div class="app-logo">
                <img src="data:image/png;base64,{encoded_image}">
            </div>
            """,
            unsafe_allow_html=True
        )
    # App Header
    st.markdown("""
    <style>
    .title-container .title {
        font-size: 2.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-shadow: 
            0 2px 4px rgba(0, 0, 0, 0.3),
            0 0 20px rgba(255, 255, 255, 0.2);
        letter-spacing: 0.8px;
        line-height: 1.3;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        color: white;
        padding: 0.5rem 0;
        text-align: center;
    }
    </style>
    
    <div class="title-container">
        <div class="title">Advanced AI-powered detection of Gallbladder Diseases from ultrasound medical image</div>
    </div>
""", unsafe_allow_html=True)

    # Load model and encoder
    model_path = os.path.join(os.getcwd(), "models", "Gallbladder31.pth")  # Assuming models folder in the root directory
    encoder_path = os.path.join(os.getcwd(), "models", "encoderr.pkl") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(encoder_path)
    num_classes = len(encoder.classes_)
    model = load_model(model_path, num_classes, device)

    mean = torch.tensor([0.2765, 0.2770, 0.2767]).to(device)
    std = torch.tensor([0.2152, 0.2151, 0.2159]).to(device)

    # Tab interface
    with st.container():
        tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Capture Image"])

        with tab1:
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], 
                                          label_visibility="collapsed")
            
            st.markdown("</div>", unsafe_allow_html=True)

            if uploaded_file is not None:
                try:
                    file_bytes = uploaded_file.read()
                    if not file_bytes:
                        st.error("Uploaded file is empty.")
                        return

                    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.image(image, channels="BGR", use_container_width=True, output_format="JPEG")
                        st.markdown('<p style="color:blue; font-weight:bold;">Uploaded Image</p>', unsafe_allow_html=True)
                    
                    with col2:
                        with st.spinner("Analyzing image..."):
                            image_tensor = preprocess_numpy_image(image, mean, std, device)
                            predicted_label, probabilities, predicted_class_index = predict_image(model, image_tensor, encoder, device)
                            
                            st.markdown(f"""
                                <div class="result-box">
                                    <h4 style="color: var(--primary); margin-bottom: 1rem;">
                                        <span class="disease-icon">🩺</span> Diagnosis Result
                                    </h4>
                                    <p style="font-size: 1.2rem; font-weight: 600; color: var(--dark);">
                                        {predicted_label}
                                    </p>
                                    <p style="font-size: 1rem; margin-top: 0.5rem;">
                                        Confidence: <strong>{probabilities[0][predicted_class_index].item():.1%}</strong>
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Expandable section for detailed confidence levels
                            with st.expander("Show detailed confidence levels for all classes", expanded=False):
                                st.markdown("<h4 style='margin-top: 0;'>Detailed Confidence Levels</h4>", 
                                            unsafe_allow_html=True)
                                
                                # Sort classes by probability in descending order
                                sorted_indices = torch.argsort(probabilities[0], descending=True)
                                
                                for idx in sorted_indices:
                                    class_name = encoder.classes_[idx]
                                    prob = probabilities[0][idx].item()
                                    
                                    # Highlight the predicted class
                                    if idx == predicted_class_index:
                                        bar_color = "var(--success)"
                                        text_style = "font-weight: 600; color: var(--success);"
                                    else:
                                        bar_color = "var(--primary)"
                                        text_style = ""
                                        
                                    st.markdown(f"""
                                        <div style="margin-bottom: 0.5rem;">
                                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                                                <span style="{text_style}">{class_name}</span>
                                                <span style="{text_style}">{prob:.1%}</span>
                                            </div>
                                            <div class="probability-bar">
                                                <div class="probability-fill" style="width: {prob*100}%; background: linear-gradient(90deg, {bar_color} 0%, #5dade2 100%);">
                                                    <div class="probability-label">{prob:.1%}</div>
                                                </div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"An error occurred while processing the image: {str(e)}")

        with tab2:
            st.markdown("""
                <div class="camera-box">
                    <h3 style="color: var(--primary); margin-bottom: 0.1rem;">Capture Gallbladder Image</h3>
                    <p style="margin-bottom: 0.1rem;">Use your camera to capture an image for analysis.</p>
            """, unsafe_allow_html=True)
            
            # Use Streamlit's camera input instead of WebRTC
            picture = st.camera_input("Take a picture")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            if picture is not None:
                try:
                    # Convert the picture to numpy array
                    image = cv2.imdecode(np.frombuffer(picture.getvalue(), np.uint8), 1)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.image(image, channels="BGR", caption="Captured Image", 
                               use_column_width=True, output_format="JPEG")
                    
                    with col2:
                        with st.spinner("Analyzing image..."):
                            image_tensor = preprocess_numpy_image(image, mean, std, device)
                            predicted_label, probabilities, predicted_class_index = predict_image(model, image_tensor, encoder, device)
                            
                            st.markdown(f"""
                                <div class="result-box">
                                    <h4 style="color: var(--primary); margin-bottom: 1rem;">
                                        <span class="disease-icon">🩺</span> Diagnosis Result
                                    </h4>
                                    <p style="font-size: 1.2rem; font-weight: 600; color: var(--dark);">
                                        {predicted_label}
                                    </p>
                                    <p style="font-size: 1rem; margin-top: 0.5rem;">
                                        Confidence: <strong>{probabilities[0][predicted_class_index].item():.1%}</strong>
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Expandable section for detailed confidence levels
                            with st.expander("Show detailed confidence levels for all classes", expanded=False):
                                st.markdown("<h4 style='margin-top: 0;'>Detailed Confidence Levels</h4>", 
                                            unsafe_allow_html=True)
                                
                                # Sort classes by probability in descending order
                                sorted_indices = torch.argsort(probabilities[0], descending=True)
                                
                                for idx in sorted_indices:
                                    class_name = encoder.classes_[idx]
                                    prob = probabilities[0][idx].item()
                                    
                                    # Highlight the predicted class
                                    if idx == predicted_class_index:
                                        bar_color = "var(--success)"
                                        text_style = "font-weight: 600; color: var(--success);"
                                    else:
                                        bar_color = "var(--primary)"
                                        text_style = ""
                                        
                                    st.markdown(f"""
                                        <div style="margin-bottom: 0.5rem;">
                                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                                                <span style="{text_style}">{class_name}</span>
                                                <span style="{text_style}">{prob:.1%}</span>
                                            </div>
                                            <div class="probability-bar">
                                                <div class="probability-fill" style="width: {prob*100}%; background: linear-gradient(90deg, {bar_color} 0%, #5dade2 100%);">
                                                    <div class="probability-label">{prob:.1%}</div>
                                                </div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred while processing the captured image: {str(e)}")

    # Footer
    st.markdown("""
        <div class="footer">
            <p>© 2025 Gallbladder Disease Diagnosis System | Developed with Streamlit and PyTorch</p>
            <p style="font-size: 0.8rem;">For medical professionals only. Not for self-diagnosis.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
