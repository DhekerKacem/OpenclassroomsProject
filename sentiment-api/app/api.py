import os

# Définition du chemin vers le modèle
model_path = 'C:/Users/Dhaker/Documents/OpenclassroomsProject/model_bert'

# Vérifier si le chemin existe
if os.path.exists(model_path):
    print("Le dossier existe.")
    # Lister tous les fichiers dans le dossier
    print("Liste des fichiers et dossiers dans le chemin spécifié:")
    files = os.listdir(model_path)
    for file in files:
        print(file)
else:
    print("Le dossier n'existe pas, vérifiez le chemin d'accès.")

from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

app = Flask(__name__)

# Chargement du modèle et du tokenizer
model_path = 'C:/Users/Dhaker/Documents/OpenclassroomsProject/model_bert'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
model.eval()  # Mettre le modèle en mode évaluation

# Ajout d'une couche linéaire pour obtenir les logits
linear_layer = nn.Linear(model.config.hidden_size, 1)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['tweet']
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=200,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = linear_layer(pooled_output)

    prediction = torch.sigmoid(logits).item()
    print(prediction)
    return jsonify({'sentiment': 'Positive' if prediction >= 0.5 else 'Negative'})

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True, port=8887)  # Lancer l'application


