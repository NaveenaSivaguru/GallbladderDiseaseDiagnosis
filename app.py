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

# --- Model and Helper Functions ---
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
        return predicted_class_label, probabilities

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
    st.title("Gastrointestinal Disease Classification")

    # Use relative paths to store models and encoder
    model_path = os.path.join(os.getcwd(), "models", "Gallbladder31.pth")  # Assuming models folder in the root directory
    encoder_path = os.path.join(os.getcwd(), "models", "encoderr.pkl")  # Assuming encoder is also in the models folder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = load_encoder(encoder_path)
    num_classes = len(encoder.classes_)
    model = load_model(model_path, num_classes, device)

    mean = torch.tensor([0.2765, 0.2770, 0.2767]).to(device)
    std = torch.tensor([0.2152, 0.2151, 0.2159]).to(device)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            if not file_bytes:
                st.error("Uploaded file is empty.")
                return

            image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                image_tensor = preprocess_numpy_image(image, mean, std, device)

                predicted_class, probabilities = predict_image(model, image_tensor, encoder, device)
                st.write(f"Predicted class: {predicted_class}")
                st.write("Probabilities:")
                for i, class_name in enumerate(encoder.classes_):
                    st.write(f"- {class_name}: {probabilities[0][i].item():.4f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
   
