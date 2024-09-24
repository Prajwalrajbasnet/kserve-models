import kserve

from kserve.logging import logger
from sentence_transformers import SentenceTransformer

from typing import List, Dict
from pydantic import BaseModel


class RequestSchema(BaseModel):
    is_query_to_passage: bool = False
    texts: List[str]
    normalize_embeddings: bool = True
    batch_size: int = 32


class ResponseSchema(BaseModel):
    embeddings: List[float]


class EmbeddingModel(kserve.Model):
    model: SentenceTransformer

    def __init__(self, name: str):
        super().__init__(name)

        self.model = None
        self.ready = False

        self.load()

    # Load the model image
    def load(self):
        # TODO: Custom code to resolve path for the latest/required version of the model from model registry
        # TODO: Use cuda if appropriate
        logger.info("Starting to load model...")
        model_path = "BAAI/bge-large-en-v1.5"
        self.model = SentenceTransformer(model_name_or_path=model_path)
        self.ready = True
        logger.info("Model loaded to memory...")

    def predict(
        self,
        input: Dict,
        _: Dict[str, str] = None,
    ) -> Dict:
        payload = RequestSchema(**input)
        
        instruction = "Represent this sentence for searching relevant passages:"
        texts = payload.texts

        if payload.is_query_to_passage:
            texts = [instruction + text for text in texts]

        embeddings = self.model.encode(
            texts, normalize_embeddings=payload.normalize_embeddings, batch_size=payload.batch_size
        )

        return ResponseSchema(
            embeddings=embeddings,
        ).model_dump()


if __name__ == "__main__":
    model = EmbeddingModel("embedding-model")
    kserve.ModelServer().start([model])
