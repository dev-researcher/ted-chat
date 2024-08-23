# from google.cloud import aiplatform
# from config import PROJECT_ID, REGION, MODEL_NAME

# def chat_with_model(prompt):
#     client = aiplatform.gapic.PredictionServiceClient()
#     endpoint = f"projects/mychat-433121/locations/us-central1/endpoints/publishers/google/models/gemini-pro"
    
#     response = client.predict(
#         endpoint=endpoint,
#         instances=[{"content": prompt}],
#         parameters={}
#     )
#     return response.predictions[0]["content"]

# if __name__ == "__main__":
#     prompt = input("Say something: ")
#     response = chat_with_model(prompt)
#     print(f"Model response: {response}")

import pandas as pd
from google.cloud import aiplatform
from vertexai import init
from vertexai.generative_models import GenerativeModel, ChatSession
from config import PROJECT_ID, REGION, MODEL_NAME
from transformers import BertTokenizer, BertModel
import torch

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

def interact_with_user(user_input):
    # Aquí buscarías en los embeddings la respuesta más relevante
    response = your_model_function(user_input, df)
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
