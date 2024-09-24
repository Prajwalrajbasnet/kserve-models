import json
from typing import Dict

from sentence_transformers import SentenceTransformer

from cog import BasePredictor, Input


class Predictor(BasePredictor):
    model: SentenceTransformer

    def setup(self):
        """Load and initialize the model"""
        model_path = "./models/BAAI_bge-large-en"
        self.model = SentenceTransformer(model_name_or_path=model_path)

    # Define the arguments and types the model takes as input
    def predict(self, texts: str = Input(description='text to embed, formatted as JSON list of strings (e.g. ["hello", "world"])', default=""),) -> Dict:
        """Run a single prediction on the model"""
        texts = json.loads(texts)
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        return {
            "embeddings": embeddings,
        }
