from flask import Flask, request, jsonify
import torch
import pickle
from torchvision import transforms
from PIL import Image
from src.model import get_model
from src.utils.output_utils import prepare_output
from src.args import get_parser
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import requests
from flask_cors import CORS 

app = Flask(__name__)
CORS(app) 

# Configuration
MODEL_URL = "https://dl.fbaipublicfiles.com/inversecooking/modelbest.ckpt"
MODEL_PATH = "/tmp/modelbest.ckpt"  # Use /tmp directory which is writable in Vercel
data_dir = os.path.join('./', 'data')

# Global variables
model = None
ingr_vocab = None
instr_vocab = None
transform = None
device = None

def load_model():
    global model, ingr_vocab, instr_vocab, transform, device
    
    if model is not None:
        return
        
    # Load vocabularies
    ingr_vocab = pickle.load(open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb'))
    instr_vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))
    
    # Set device
    device = torch.device('cpu')
    
    # Initialize model
    args = get_parser()
    args.maxseqlen = 15
    args.ingrs_only = False
    model = get_model(args, len(ingr_vocab), len(instr_vocab))
    
    # Download and load model
    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.to(device)
    model.eval()
    
    # Initialize transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        load_model()
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model.sample(image_tensor, greedy=True, temperature=1.0, beam=-1, true_ingrs=None)

        ingr_ids = outputs['ingr_ids'].cpu().numpy()
        recipe_ids = outputs['recipe_ids'].cpu().numpy()

        outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingr_vocab, instr_vocab)

        if not valid['is_valid']:
            return jsonify({
                'error': 'Invalid recipe generated',
                'reason': valid['reason']
            }), 400

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        response = {
            'title': outs['title'],
            'ingredients': outs['ingrs'],
            'instructions': outs['recipe'],
            'status': 'success'
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/')
def home():
    return "Hello, Flask running in Vercel!"


if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))
    # Get host from environment variable or use default
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port)

