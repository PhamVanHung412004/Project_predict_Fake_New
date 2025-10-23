import torch
import torch.nn as nn
import pickle
import re
import string
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from torchtext.data.utils import get_tokenizer
import os
from gemini_service import gemini_service

app = Flask(__name__)
CORS(app)

# Load model và vocabulary
model = None
vocabulary = None
tokenizer = None
device = torch.device("cpu")

# TextClassificationModel class (từ notebook)
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, inputs, offsets):
        embedded = self.embedding(inputs, offsets)
        return self.fc(embedded)

# Hàm preprocess text (từ notebook)
def preprocess_text(text):
    url_pattern = re.compile(r'https?://\s+\wwww\.\s+')
    text = url_pattern.sub(r" ", text)

    html_pattern = re.compile(r'<[^<>]+>')
    text = html_pattern.sub(" ", text)

    replace_chars = list(string.punctuation + string.digits)
    for char in replace_chars:
        text = text.replace(char, " ")

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r" ", text)

    text = " ".join(text.split())
    return text.lower()

def load_model():
    global model, vocabulary, tokenizer
    
    try:
        # Load vocabulary
        with open('model/vocab.pkl', 'rb') as f:
            vocabulary = pickle.load(f)
        
        # Load model
        vocab_size = len(vocabulary)
        embed_dim = 100
        num_class = 2  # 0: Giả mạo, 1: Bình thường
        
        model = TextClassificationModel(vocab_size, embed_dim, num_class)
        model.load_state_dict(torch.load('model/model_state.pth', map_location=device))
        model.eval()
        
        # Initialize tokenizer
        tokenizer = get_tokenizer("basic_english")
        
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_text(text):
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Tokenize
        tokens = tokenizer(processed_text)
        
        # Convert to indices
        encoded = torch.tensor(vocabulary(tokens), dtype=torch.int64)
        
        # Make prediction
        with torch.no_grad():
            output = model(encoded, torch.tensor([0]))
            prediction = output.argmax(1).item()
            confidence = torch.softmax(output, dim=1).max().item()
        
        return prediction, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        include_analysis = data.get('include_analysis', True)  # New parameter
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        prediction, confidence = predict_text(text)
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        labels = {0: "Giả mạo", 1: "Bình thường"}
        result = {
            'prediction': prediction,
            'label': labels[prediction],
            'confidence': round(confidence * 100, 2),
            'text': text
        }
        
        # Add Gemini analysis if requested
        if include_analysis:
            try:
                gemini_analysis = gemini_service.analyze_prediction(
                    text=text,
                    prediction=prediction,
                    confidence=result['confidence'],
                    label=result['label']
                )
                result['analysis'] = gemini_analysis
            except Exception as e:
                print(f"Gemini analysis failed: {e}")
                result['analysis'] = {
                    'success': False,
                    'error': 'Analysis service unavailable',
                    'source': 'error'
                }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Load model when starting the app
    if load_model():
        print("Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Exiting...")
