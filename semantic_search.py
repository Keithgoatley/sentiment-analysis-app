import faiss
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []

    def build_index(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.texts = texts

    def query(self, text, top_k=5):
        query_embedding = self.model.encode([text], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)
        results = [(self.texts[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results
