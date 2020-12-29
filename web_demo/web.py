from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from torch import load, device, cuda
from flask import Flask, request, render_template, jsonify
from predict import get_prediction_with_single
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    # sentence is received by ajax from html
    sentence = request.get_json(force=True)   

    # Predict with a single sentence
    prediction = get_prediction_with_single(tokenizer, model, sentence['text'], device)
    map_en = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'sad'}
    result = {'text': map_en[prediction.item()]}

    return jsonify(result)


# main
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICE'] = '5'

    # Load bert model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    PRETRAINED_MODEL_NAME = 'best_model_state.bin'
    state_dict = load(PRETRAINED_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4, state_dict=state_dict)
    # Device
    device = device('cuda:5' if cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)

    app.run(host='localhost', port=5000)
