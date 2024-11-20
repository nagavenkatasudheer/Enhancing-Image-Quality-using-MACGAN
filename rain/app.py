#####streamlit run app.py --server.port 8000


import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import streamlit as st
from models import MODEL_PA  # Assuming this is the correct import statement

# Define function to perform image dehazing
def dehaze_image(input_img, net):
    input_img_ = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(input_img)[None,::]
    with torch.no_grad():
        pred = net(input_img_)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    # Convert tensor to numpy array
    output_img_np = ts.permute(1, 2, 0).numpy()  # Assuming tensor shape is (C, H, W)
    # Rescale to 0-255 and convert to uint8
    output_img_np = (output_img_np * 255).astype(np.uint8)
    # Convert numpy array to PIL image
    output_img_pil = Image.fromarray(output_img_np)
    return output_img_pil

# Main function to run the Streamlit app
def main():
    st.title("Image Dehazing App")

    # Load the trained model
    model_dir = 'trained_models/our_model.pk'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckp = torch.load(model_dir, map_location=device)
    gps = 6
    blocks = 19
    net = MODEL_PA(gps=gps, blocks=blocks)
    net = nn.DataParallel(net)
    net.load_state_dict(ckp['model'])
    net.eval()

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        input_img = Image.open(uploaded_file)
        st.image(input_img, caption="Original Image", use_column_width=True)

        # Dehaze image and display result
        output_img = dehaze_image(input_img, net)
        st.image(output_img, caption="Dehazed Image", use_column_width=True)

if __name__ == "__main__":
    main()
