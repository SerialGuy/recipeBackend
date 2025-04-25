from flask import Flask, request, jsonify
import torch
import pickle
from torchvision import transforms
from PIL import Image
from model import get_model
from utils.output_utils import prepare_output
from args import get_parser
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from flask_cors import CORS 

app = Flask(__name__)
CORS(app) 
# Initialize model and vocabularies once when the server starts
data_dir = '../data'
ingr_vocab = pickle.load(open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb'))
instr_vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))

# Set device configuration
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

# Image transformations
transf_list_batch = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]
transform = transforms.Compose(transf_list_batch)

@app.route('/predict', methods=['POST'])
def predict():
    print("Received predict request")  # Add this
    app.logger.info("Processing image request")  # Add this
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Process image
        image = Image.open(file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Model inference
        with torch.no_grad():
            outputs = model.sample(image_tensor, greedy=True, temperature=1.0, beam=-1, true_ingrs=None)

        ingr_ids = outputs['ingr_ids'].cpu().numpy()
        recipe_ids = outputs['recipe_ids'].cpu().numpy()

        # Prepare output
        outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingr_vocab, instr_vocab)

        if not valid['is_valid']:
            return jsonify({
                'error': 'Invalid recipe generated',
                'reason': valid['reason']
            }), 400

        # Convert image to base64 for response
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
