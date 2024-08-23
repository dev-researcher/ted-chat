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

df = pd.read_csv('ted_ed.csv')

# Ejemplo de procesamiento: Filtrar columnas relevantes
df = df[['title', 'transcript']]

df['embeddings'] = df['transcript'].apply(generate_embeddings)

def initialize_model():
    init(project=PROJECT_ID, location=REGION)
    model = GenerativeModel(model_name=MODEL_NAME)
    chat = model.start_chat()
    return chat

def em_function(user_input, df):
    user_embedding = generate_embeddings(user_input)
    similarities = cosine_similarity([user_embedding], df['embeddings'].tolist())[0]
    most_similar_idx = np.argmax(similarities)
    response = df.iloc[most_similar_idx]['title']
    return response

def interact_with_user(user_input):
    # Aquí buscarías en los embeddings la respuesta más relevante
    response = em_function(user_input, df)
    return response

# def get_chat_response(chat: ChatSession, prompt: str) -> str:
#     # Envía like streaming
#     text_response = []
#     responses = chat.send_message(prompt, stream=True)
#     for chunk in responses:
#         text_response.append(chunk.text)
#     return "".join(text_response)

if __name__ == "__main__":
    # Inicia modelo
    chat = initialize_model()

    prompts = [
        # "Hello.",
        # "What are all the colors in a rainbow?",
        # "Why does it appear when it rains?"
        "de que color son las manzanas?"
    ]
    
    for prompt in prompts:
        response = get_chat_response(chat, prompt)
        print(f"Model response to '{prompt}': {response}")
