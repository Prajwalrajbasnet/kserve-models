from pydantic import BaseModel
from typing import List

class RequestSchema(BaseModel):
  is_query_to_passage: bool = False
  texts: List[str]
  normalize_embeddings: bool = True
  batch_size: int = 32

class ResponseSchema(BaseModel):
  embeddings: List[str]
