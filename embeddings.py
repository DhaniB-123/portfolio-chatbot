import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

def get_embedding(text):
    result = client.feature_extraction(text, model="sentence-transformers/all-MiniLM-L6-v2")
    return result.tolist()
