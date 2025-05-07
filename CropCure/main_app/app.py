import os
import sys
import torch
import pandas as pd
import numpy as np
import requests
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF

# Try importing CNN module properly
try:
    import CNN
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        import CNN
    except ModuleNotFoundError:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from CropCure import CNN
        except ModuleNotFoundError:
            print("ERROR: Could not import CNN module. Please check file structure.")
            class CNN:
                def __init__(self, num_classes):
                    self.num_classes = num_classes
                def __call__(self, x):
                    return np.zeros((1, self.num_classes))
                def eval(self):
                    pass
                def load_state_dict(self, state_dict):
                    pass

# For downloading from Google Drive reliably
import gdown

# Google Drive file ID for the model
file_id = '1En73N317hTlvJpZDa-FqsMsIMskzU70h'
file_name = 'datasetofdisease.pt'

# Function to download the model file using gdown
def download_model():
    if not os.path.exists(file_name):
        print(f"{file_name} not found. Downloading...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, file_name, quiet=False)
        print("Download complete.")
    else:
        print(f"{file_name} already exists.")

# Download the model if needed
download_model()

# Load the model
model = CNN.CNN(39)
model.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
model.eval()

# Load disease and supplement info
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Prediction function
def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

# Flask app
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name,
                               simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
