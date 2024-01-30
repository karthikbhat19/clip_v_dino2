from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os
import torch
import pdb

access_token = "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"

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', token=access_token)
model = AutoModel.from_pretrained('facebook/dinov2-base', token=access_token)

# change this depending on where you put the Image folder
image_folder_relative = "Images"
image_folder = os.path.join(os.getcwd(), image_folder_relative)
# For my own image folder, should be: C:\Users\Daniel\OneDrive\Desktop\MS\CSE 290D\Images

# Set the desired batch size. CURRENTLY ONLY SUPPORTING 1 AT A TIME
# If you want to increase the batch size, change how we add the images to the dictionary
batch_size = 1

# Dict to store processed outputs before extracting the hidden state
# Key: Image string
# Value: model inference on image
all_outputs = {}

# Dict to store final hidden states
final_outputs = {}

# Iterate through images in batches
for i in range(0, len(os.listdir(image_folder)), batch_size):
    batch_images = []

    # Load a batch of images
    for j in range(batch_size):
        idx = i + j
        if idx < len(os.listdir(image_folder)):
            image_str = os.listdir(image_folder)[idx]
            image_path = os.path.join(image_folder, image_str)
            opened_image = Image.open(image_path)
            batch_images.append(opened_image)

    # Process the batch of images using the image processor and obtain tensor representations
    inputs = processor(images=batch_images, return_tensors="pt")
    
    # Perform inference on the input tensor using the DINO model
    outputs = model(**inputs)

    # Store the output for later analysis or processing
    all_outputs[image_str] = outputs

# Compare outputs of the original model and traced model
for outputs in all_outputs.items():
    with torch.no_grad():
        last_hidden_states = outputs[1].last_hidden_state
        image_features1 = last_hidden_states.mean(dim=1)
        final_outputs[outputs[0]] = image_features1
        