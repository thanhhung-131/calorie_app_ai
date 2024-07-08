import os
from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import urllib.request
import requests

app = Flask(__name__)

# Define the path to the classes.txt file
classes_file_path = "project_ai\classes.txt"

classes = []

# Check if the classes.txt file exists
if os.path.exists(classes_file_path):
    # Read the content of classes.txt and split it into lines
    with open(classes_file_path, 'r') as file:
        classes = file.read().splitlines()
else:
    print("Error: classes.txt file not found!")

# Load the saved model and label encoder
model_path = "project_ai\solution.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

class Label_encoder:
    def __init__(self, labels):
        labels = list(set(labels))
        self.labels = {label: idx for idx, label in enumerate(classes)}

    def get_label(self, idx):
        return list(self.labels.keys())[idx]

    def get_idx(self, label):
        return self.labels[label]

encoder = Label_encoder(classes)

print(os.getcwd())
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

label_encoder = Label_encoder(classes)
def preprocess_image(image):
    # Resize the image to 256x256 pixels
    resized_image = image.resize((256, 256))
    # Center crop the resized image to 224x224 pixels
    cropped_image = transforms.CenterCrop(224)(resized_image)
    # Convert the image to a PyTorch tensor
    tensor_image = transforms.ToTensor()(cropped_image)
    # Normalize the image
    normalized_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_image)
    # Add a batch dimension to the image tensor
    preprocessed_image = normalized_image.unsqueeze(0)
    return preprocessed_image


# Define the route for the home page
@app.route('/')
def home():
  return render_template('index.html')

# Define the route for image upload and prediction
@app.route('/predict', methods=['POST'])


def predict():
    try:
        # Get the uploaded image data (either file or URL)
        if 'image_url' in request.form:
            # If image URL is provided, use it
            image_url = request.form['image_url']
            # Download image data from the URL
            image_data = urllib.request.urlopen(image_url)
            # Open the image from the downloaded data
            image = Image.open(image_data)
        elif 'image_file' in request.files:
            # If image file is provided, read it
            image_file = request.files['image_file']
            # Open the image file
            image = Image.open(image_file)
        else:
            return {'status': 'error', 'message': 'No image provided'}, 400

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Perform prediction
        with torch.no_grad():
            model.eval()
            output = model(preprocessed_image.to(device))

        # Get predicted class index and probability
        _, predicted_idx = torch.max(output, 1)
        predicted_idx = predicted_idx.item()
        predicted_prob = torch.softmax(output, dim=1)[0, predicted_idx].item()

        # Map index to class name
        predicted_label = label_encoder.get_label(predicted_idx)

        # Check if accuracy is below 70%
        if predicted_prob < 0.7:
            return {'status': 'low_accuracy', 'message': 'Unable to recognize'}

        # If prediction is successful, call the API to get food information
        food_name = predicted_label.lower()
        response = requests.get(f'http://34.203.34.254:3000/api/foods/search-one?name={food_name}')
        if response.status_code == 200:
            food_info = response.json()
            return {'status': 'success', 'prediction': predicted_label, 'accuracy': predicted_prob, 'food_info': food_info}
        else:
            return {'status': 'error', 'message': 'Failed to get food information'}, 500

    except Exception as e:
        # Return an error message if prediction fails
        error_message = str(e)
        return {'status': 'error', 'message': error_message}, 500


if __name__ == '__main__':
  app.run()
