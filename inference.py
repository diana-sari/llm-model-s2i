from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("model", num_labels=4)

def preprocess(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    return inputs

def postprocess(predictions):
    predicted_labels = np.argmax(predictions.logits, axis=-1)
    return predicted_labels.tolist()

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    inputs = preprocess(text)
    predictions = model(**inputs)
    output = postprocess(predictions)
    return jsonify({'predictions': output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
