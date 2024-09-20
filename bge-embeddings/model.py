import kserve

from sentence_transformers import SentenceTransformer

from .schemas import RequestSchema, ResponseSchema


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
        model_path = "./models/BAAI_bge-large-en"
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
