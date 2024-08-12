import os
os.environ['TRANSFORMERS_CACHE'] = '/cache/'
os.environ['HF_HOME'] = '/cache/'


import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from google.cloud import storage
import sys


# Dynamically change model: based on command line argument
args = sys.argv
dynamic_model = "compot_model_v2.bin"
if len(args) > 1:
    dynamic_model = args[1]


# download model from GCS
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'com-sci-2-xxxx.json' # point to json file w/ credentials
storage_client = storage.Client()
bucket = storage_client.get_bucket('canonical_com_sci_2')
blob = bucket.blob('models/' + dynamic_model)
model_location = '/tmp/' + dynamic_model
blob.download_to_filename(model_location)

# init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', return_dict=False)
        self.drop = nn.Dropout(p=0)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = SentimentClassifier(2)
model.load_state_dict(torch.load(model_location, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')), strict=False)
model.eval()


# server
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        headers = dict(request.headers)
        if not request.is_json:
            return jsonify({'error': 'Request must be in JSON format', 'headers': headers})
        
        data = request.json
        instances = data['instances']

        if not instances:
            return jsonify({'error': 'Please provide instances for prediction.'})
        
        predictions = []
        
        for instance in instances:
            encoded_review = tokenizer.encode_plus(
                instance,
                max_length=512,
                add_special_tokens=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids = encoded_review['input_ids']
            attention_mask = encoded_review['attention_mask']

            with torch.no_grad():
                output = model(
                    input_ids.to(device),
                    attention_mask.to(device)
                )
            
            _, prediction = torch.max(output, dim=1)
            prob_infer = F.softmax(output, dim=1)
            predictions.append(float(prob_infer.cpu().detach().numpy().T[1][0]))

        return jsonify({
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

# create dummy /test endpoint to check if server is running
@app.route('/test', methods=['GET'])
def test():
    return jsonify({})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
