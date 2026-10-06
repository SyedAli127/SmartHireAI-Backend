# embedder.py
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Embedder:
    def __init__(self, model="text-embedding-3-small"):
        self.model = model

    def embed_text(self, text: str):
        text = text.replace("\n", " ")
        resp = client.embeddings.create(
            model=self.model,
            input=text
        )
        return resp.data[0].embedding
