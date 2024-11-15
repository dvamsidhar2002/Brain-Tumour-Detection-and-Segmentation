# Vision Transformer + SVM Classifier [CLAHE Processed Images]

# DATA PREPARATION

import os
import cv2

def apply_clahe(image):
    """
    Apply CLAHE to the input grayscale image.

    Parameters:
    - image: Grayscale image

    Returns:
    - CLAHE-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def preprocess_image(image_path):
    """
    Load and preprocess the image for CLAHE enhancement.

    Parameters:
    - image_path: Path to the input image

    Returns:
    - Grayscale preprocessed image
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (256, 256))  # Resize to 256x256
    return gray_image

def save_image(image, save_path):
    """
    Save the preprocessed image.

    Parameters:
    - image: CLAHE-enhanced grayscale image
    - save_path: Path to save the image
    """
    cv2.imwrite(save_path, image)

def process_and_save_clahe_images(input_dir, output_dir):
    categories = ['yes', 'no']
    for category in categories:
        input_folder = os.path.join(input_dir, category)
        output_folder = os.path.join(output_dir, category)
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg"):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)

                # Load and preprocess the image
                gray_image = preprocess_image(input_path)

                # Apply CLAHE
                clahe_image = apply_clahe(gray_image)

                # Save the CLAHE-enhanced image
                save_image(clahe_image, output_path)
                print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_dir = "F:/Brain Tumor Detection/Dataset/Training"
    output_dir = "F:/Brain Tumor Detection/Image Preprocessing Experiments/CLAHE/Training"
    process_and_save_clahe_images(input_dir, output_dir)

import os
import cv2

def apply_clahe(image):
    """
    Apply CLAHE to the input grayscale image.

    Parameters:
    - image: Grayscale image

    Returns:
    - CLAHE-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def preprocess_image(image_path):
    """
    Load and preprocess the image for CLAHE enhancement.

    Parameters:
    - image_path: Path to the input image

    Returns:
    - Grayscale preprocessed image
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (256, 256))  # Resize to 256x256
    return gray_image

def save_image(image, save_path):
    """
    Save the preprocessed image.

    Parameters:
    - image: CLAHE-enhanced grayscale image
    - save_path: Path to save the image
    """
    cv2.imwrite(save_path, image)

def process_and_save_clahe_images(input_dir, output_dir):
    categories = ['yes', 'no']
    for category in categories:
        input_folder = os.path.join(input_dir, category)
        output_folder = os.path.join(output_dir, category)
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg"):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)

                # Load and preprocess the image
                gray_image = preprocess_image(input_path)

                # Apply CLAHE
                clahe_image = apply_clahe(gray_image)

                # Save the CLAHE-enhanced image
                save_image(clahe_image, output_path)
                print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_dir = "F:/Brain Tumor Detection/Dataset/Testing"
    output_dir = "F:/Brain Tumor Detection/Image Preprocessing Experiments/CLAHE/Testing"
    process_and_save_clahe_images(input_dir, output_dir)


# Training and Performance Report Vision Transformer

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from torchvision import transforms
from torchvision.models import vit_b_16
from PIL import Image
from tqdm import tqdm

# Paths to the HOG preprocessed images
data_dir = 'F:/Brain Tumor Detection/Image Preprocessing Experiments/CLAHE/Training'
yes_folder = os.path.join(data_dir, 'yes')
no_folder = os.path.join(data_dir, 'no')


# Function to load images from folder and return them as a list
def load_images_from_folder(folder, label, transform=None):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')  # Open the image as RGB
        if transform:
            img = transform(img)  # Apply transformations (e.g., normalization)
        images.append(img)
        labels.append(label)
    return images, labels


# Define the transformation for image preprocessing before passing to ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ViT input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pre-trained ViT
])

# Load the images
yes_images, yes_labels = load_images_from_folder(yes_folder, 1, transform)
no_images, no_labels = load_images_from_folder(no_folder, 0, transform)

# Combine the images and labels
images = yes_images + no_images
labels = yes_labels + no_labels

# Stack images into a tensor and labels into a numpy array
images_tensor = torch.stack(images)
labels = np.array(labels)

# Load pre-trained Vision Transformer (ViT) model
vit_model = vit_b_16(pretrained=True)
vit_model.eval()  # Set the model to evaluation mode

# Move the model to GPU if available (if not, will run on CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)


# Function to extract features from ViT model with batching
def extract_features(images_tensor, model, batch_size=32):
    all_features = []
    num_batches = len(images_tensor) // batch_size + (1 if len(images_tensor) % batch_size != 0 else 0)

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(images_tensor))
        batch = images_tensor[start_idx:end_idx].to(device)  # Move batch to GPU if available

        with torch.no_grad():  # No need to compute gradients for feature extraction
            outputs = model(batch)  # Pass through ViT
            all_features.append(outputs.cpu().numpy())  # Move features to CPU

    return np.concatenate(all_features, axis=0)


# Extract features from the images in smaller batches
features = extract_features(images_tensor, vit_model, batch_size=32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


