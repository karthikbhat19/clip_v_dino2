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

# Set the desired batch size
batch_size = 1

# List to store processed outputs before extracting the hidden state
all_outputs = []

# List to store final hidden states
final_outputs = []

# Iterate through images in batches
for i in range(0, len(os.listdir(image_folder)), batch_size):
    batch_images = []

    # Load a batch of images
    for j in range(batch_size):
        idx = i + j
        if idx < len(os.listdir(image_folder)):
            image_path = os.path.join(image_folder, os.listdir(image_folder)[idx])
            opened_image = Image.open(image_path)
            batch_images.append(opened_image)

    # Process the batch of images using the image processor and obtain tensor representations
    inputs = processor(images=batch_images, return_tensors="pt")
    
    # Perform inference on the input tensor using the DINO model
    outputs = model(**inputs)

    # Store the output for later analysis or processing
    all_outputs.append(outputs)
    if i == 2:
        break

# pdb.set_trace() 
# Compare outputs of the original model and traced model
for outputs in all_outputs:
    last_hidden_states = outputs.last_hidden_state
    image_features1 = last_hidden_states.mean(dim=1)
    final_outputs.append(image_features1)
    
    # We have to force return_dict=False for tracing
    model.config.return_dict = False

    # Indicates that this block code doesn't track gradients, which is typical when performing
    # inferences or evaluating a model that doesn't need to compute gradients for backpropagation
    with torch.no_grad():
        # Create a traced version of the original model, which improves the efficiency of model deployment,
        # because traced models can be more lightweight and offer faster inference performances
        traced_model = torch.jit.trace(model, [inputs.pixel_values])
        # Use the traced model to perform inference on the input tensor and store the output in traced_outputs
        traced_outputs = traced_model(inputs.pixel_values)

    # Calculates the maximum absolute difference between the original model's output and the traced model's output,
    # and if this is small, their outputs closely match, validating the accuracy of the tracing process
    print((last_hidden_states - traced_outputs[0]).abs().max())

    print("Shape of image feature dimension: ", image_features1.shape)
    print("Shape of the output tensor: ", traced_outputs[0].shape)
    print("Raw output tensor values: ", traced_outputs[0])
