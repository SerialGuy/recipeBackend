import streamlit as st
import torch
import pickle
from torchvision import transforms
from PIL import Image
from model import get_model
from utils.output_utils import prepare_output
from args import get_parser
import time
import os
import random
import matplotlib.pyplot as plt
from io import BytesIO
import requests

# Load vocabularies and model weights
data_dir = '../data'  # Path to your data directory
ingr_vocab = pickle.load(open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb'))
instr_vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))

ingr_vocab_size = len(ingr_vocab)
instrs_vocab_size = len(instr_vocab)
output_dim = instrs_vocab_size

# Set device to GPU or CPU
use_gpu = False
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'

# Initialize model
args = get_parser()
args.maxseqlen = 15
args.ingrs_only = False
model = get_model(args, len(ingr_vocab), len(instr_vocab))
model_path = os.path.join(data_dir, 'modelbest.ckpt')
model.load_state_dict(torch.load(model_path, map_location=map_loc))
model.to(device)
model.eval()

# Define image transformation for inference
transf_list_batch = [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
to_input_transf = transforms.Compose(transf_list_batch)

# Streamlit UI
st.title("Image-to-Recipe Generation")
st.write("Upload an image of food to generate a recipe:")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert('RGB')

    # Preprocess the image
    transf_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    transform = transforms.Compose(transf_list)
    image_transf = transform(image)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)

    # Display the image
    st.image(image_transf, caption="Uploaded Image", use_column_width=True)

    # Model inference
    with torch.no_grad():
        outputs = model.sample(image_tensor, greedy=True, temperature=1.0, beam=-1, true_ingrs=None)

    ingr_ids = outputs['ingr_ids'].cpu().numpy()
    recipe_ids = outputs['recipe_ids'].cpu().numpy()

    # Post-process and prepare the output
    outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingr_vocab, instr_vocab)

    if valid['is_valid']:
        st.subheader("Generated Recipe:")

        st.markdown("### Title:")
        st.write(outs['title'])

        st.markdown("### Ingredients:")
        st.write(', '.join(outs['ingrs']))

        st.markdown("### Instructions:")
        st.write('\n'.join(outs['recipe']))

    else:
        st.error("Not a valid recipe!")
        st.write(f"Reason: {valid['reason']}")
