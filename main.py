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

from google.cloud import aiplatform
from vertexai import init
from vertexai.generative_models import GenerativeModel, ChatSession
from config import PROJECT_ID, REGION, MODEL_NAME

def initialize_model():
    init(project=PROJECT_ID, location=REGION)
    model = GenerativeModel(model_name=MODEL_NAME)
    chat = model.start_chat()
    return chat

def get_chat_response(chat: ChatSession, prompt: str) -> str:
    # Envía like streaming
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)

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
