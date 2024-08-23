import pandas as pd
from google.cloud import aiplatform
from vertexai import init
from vertexai.generative_models import GenerativeModel, ChatSession
from config import PROJECT_ID, REGION, MODEL_NAME
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Cargar modelo y tokenizer de BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def generate_embeddings(text):
    # Tokenizar el texto
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    # Generar embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    # Usar la salida de la última capa oculta
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()

def em_function(user_input):
    user_embedding = generate_embeddings(user_input)
    similarities = cosine_similarity([user_embedding], df['embeddings'].tolist())[0]
    most_similar_idx = np.argmax(similarities)
    response = df.iloc[most_similar_idx]['Title'] + ", you can watch the video here: " + df.iloc[most_similar_idx]['Link']
    #print(similarities)
    #print(most_similar_idx)
    return response

df = pd.read_csv('teded100.csv')

df = df[['Title', 'Caption', 'Link']]

df['embeddings'] = df['Caption'].apply(generate_embeddings)

def initialize_model():
    init(project=PROJECT_ID, location=REGION)
    model = GenerativeModel(model_name=MODEL_NAME)
    chat = model.start_chat()
    return chat

def interact_with_user(user_input):
    # Aquí buscarías en los embeddings la respuesta más relevante
    response = em_function(user_input)
    return response

if __name__ == "__main__":
    # Inicia modelo
    chat = initialize_model()
    while(True):
        user_prompt = input("¿Qué es lo que te gustaría aprender? o escribe exit para salir: ")
        if(user_prompt.lower() == "exit"):
            break
        else:
            response = interact_with_user(user_prompt)
            print(f"Respuesta del modelo: {response}")