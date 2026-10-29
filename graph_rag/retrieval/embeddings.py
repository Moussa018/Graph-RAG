import numpy as np
from google.genai import types

BATCH_SIZE = 100


class GeminiEmbedder:
    """Embeddings Gemini, avec des task types distincts pour les documents et les questions."""

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def _embed(self, texts, task_type):
        vectors = []
        for i in range(0, len(texts), BATCH_SIZE):
            response = self.client.models.embed_content(
                model=self.model,
                contents=texts[i:i + BATCH_SIZE],
                config=types.EmbedContentConfig(task_type=task_type),
            )
            vectors.extend(np.asarray(emb.values, dtype=np.float32) for emb in response.embeddings)
        return vectors

    def embed_documents(self, texts):
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]
