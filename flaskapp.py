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
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

# Configuration
MODEL_URL = "https://dl.fbaipublicfiles.com/inversecooking/modelbest.ckpt"
MODEL_PATH = os.path.join(tempfile.gettempdir(), 'modelbest.ckpt')  # Use temp directory
data_dir = os.path.join('./', 'data')

# Global variables that will be initialized on first request
model = None
ingr_vocab = None
instr_vocab = None
transform = None
device = None
initialization_error = None

def download_model_with_progress(url, destination):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB chunks
        
        logger.info(f"Downloading model to {destination}")
        with open(destination, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
        logger.info("Model download completed")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

def initialize_model():
    global model, ingr_vocab, instr_vocab, transform, device, initialization_error
    
    if model is not None:  # Already initialized
        return
        
    try:
        logger.info("Starting model initialization...")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        logger.info("Created data directory")
        
        # Load vocabularies
        logger.info("Loading vocabularies...")
        ingr_vocab = pickle.load(open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb'))
        instr_vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))
        logger.info("Vocabularies loaded")
        
        # Set device configuration
        use_gpu = False
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Initialize model
        logger.info("Initializing model architecture...")
        args = get_parser()
        args.maxseqlen = 15
        args.ingrs_only = False
        model = get_model(args, len(ingr_vocab), len(instr_vocab))
        
        # Download model if needed
        if not os.path.exists(MODEL_PATH):
            if not download_model_with_progress(MODEL_URL, MODEL_PATH):
                raise Exception("Failed to download model")
            
        # Load model weights
        logger.info("Loading model weights...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=map_loc))
        model.to(device)
        model.eval()
        logger.info("Model loaded and ready")
        
        # Initialize image transformations
        transf_list_batch = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
        transform = transforms.Compose(transf_list_batch)
        logger.info("Initialization complete")
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        initialization_error = str(e)
        raise

@app.before_first_request
def before_first_request():
    try:
        initialize_model()
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")

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
    if initialization_error:
        return jsonify({
            'status': 'error',
            'message': f'Initialization failed: {initialization_error}'
        }), 500
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

