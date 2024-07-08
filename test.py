import os
import random
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

# Define the path to the classes.txt file
classes_file_path = "D:\Learning\DATN\project_ai\classes.txt"

# Load classes
if os.path.exists(classes_file_path):
    with open(classes_file_path, 'r') as file:
        classes = file.read().splitlines()
else:
    raise FileNotFoundError("Error: classes.txt file not found!")

# Define the path to the model file
model_path = "D:\Learning\DATN\project_ai\classes.txt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Label_encoder class
class Label_encoder:
    def __init__(self, labels):
        labels = list(set(labels))
        self.labels = {label: idx for idx, label in enumerate(labels)}

    def get_label(self, idx):
        return list(self.labels.keys())[idx]

    def get_idx(self, label):
        return self.labels[label]

encoder = Label_encoder(classes)

# Load the model
model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
model = model.to(device)
model = models.densenet201(weights=None)
classifier = nn.Sequential(
  nn.Linear(1920, 1024),
  nn.LeakyReLU(),
  nn.Linear(1024, 101),
)
model.classifier = classifier
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Function to preprocess image
def preprocess_image(image):
    resized_image = image.resize((256, 256))
    cropped_image = transforms.CenterCrop(224)(resized_image)
    tensor_image = transforms.ToTensor()(cropped_image)
    normalized_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_image)
    preprocessed_image = normalized_image.unsqueeze(0)
    return preprocessed_image

# Function to get random images from a directory
def get_random_images(image_dir, num_images=20):
    image_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('jpg', 'jpeg', 'png'))]
    random_images = random.sample(image_files, num_images)
    return random_images

# Function to predict image
def predict_image(image_path):
    image = Image.open(image_path)
    preprocessed_image = preprocess_image(image)
    with torch.no_grad():
        output = model(preprocessed_image.to(device))
    _, predicted_idx = torch.max(output, 1)
    predicted_idx = predicted_idx.item()
    predicted_prob = torch.softmax(output, dim=1)[0, predicted_idx].item()
    predicted_label = encoder.get_label(predicted_idx)
    return predicted_label, predicted_prob

# Main script
image_dir = 'images'
random_images = get_random_images(image_dir)
predictions = []

for image_path in random_images:
    label, prob = predict_image(image_path)
    predictions.append((image_path, label, prob))

# Display results
for image_path, label, prob in predictions:
    print(f"Image: {image_path}")
    print(f"Predicted Label: {label}")
    print(f"Probability: {prob:.2f}")
    print("----------------------")
