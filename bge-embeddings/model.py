import kserve
from sentence_transformers import SentenceTransformer

from typing import List
from pydantic import BaseModel


class RequestSchema(BaseModel):
    is_query_to_passage: bool = False
    texts: List[str]
    normalize_embeddings: bool = True
    batch_size: int = 32


class ResponseSchema(BaseModel):
    embeddings: List[str]


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
        model_path = "BAAI/bge-large-en-v1.5"
        self.model = SentenceTransformer(model_name_or_path=model_path)
        self.ready = True

    def predict(self, request: RequestSchema) -> ResponseSchema:
        embeddings = self.model.encode(request.texts, normalize_embeddings=True)

        return ResponseSchema(
            embeddings=embeddings,
        )


if __name__ == "__main__":
    model = EmbeddingModel("embedding-model")
    kserve.ModelServer().start([model])
