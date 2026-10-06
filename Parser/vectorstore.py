# vectorstore.py
import faiss
import numpy as np
import pickle
import os
import openai

class VectorStore:
    def __init__(self, persist_path="faiss_index.pkl", embedding_dim=1536, model="text-embedding-3-small", reset=False):
        self.persist_path = persist_path
        self.embedding_dim = embedding_dim
        self.model = model
        self.metadata_store = {}

        if reset or not os.path.exists(self.persist_path):
            # Fresh index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self._id_counter = 0
            self._save()
            print(f"[VectorStore] Created new FAISS index with dim {self.embedding_dim}")
        else:
            with open(self.persist_path, "rb") as f:
                self.index, self.metadata_store = pickle.load(f)
            self._id_counter = self.index.ntotal if hasattr(self.index, 'ntotal') else 0
            print(f"[VectorStore] Loaded FAISS index with {self.index.ntotal} vectors")

    # -----------------------------
    # Generate embedding
    # -----------------------------
    def embed_text(self, text):
        """Get embedding vector for text using OpenAI."""
        resp = openai.embeddings.create(
            input=text,
            model=self.model
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return vec

    # -----------------------------
    # Add document
    # -----------------------------
    def add_document(self, text, metadata=None):
        vec = self.embed_text(text).reshape(1, -1)
        self.index.add(vec)
        self.metadata_store[self._id_counter] = metadata or {}
        self._id_counter += 1
        self._save()

    # -----------------------------
    # Search / similarity
    # -----------------------------
    def search(self, text, top_k=5):
        if self.index.ntotal == 0:
            return []

        query_vec = self.embed_text(text).reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata_store):
                continue
            md = self.metadata_store.get(idx, {}).copy()
            md["similarity"] = 1 / (1 + dist)  # simple similarity score
            md["keywords"] = list(set(text.split()) & set(md.get("text", "").split()))
            results.append({"metadata": md})
        return results

    # -----------------------------
    # Persist index & metadata
    # -----------------------------
    def _save(self):
        with open(self.persist_path, "wb") as f:
            pickle.dump((self.index, self.metadata_store), f)

    # -----------------------------
    # Reset FAISS index
    # -----------------------------
    def reset(self):
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata_store = {}
        self._id_counter = 0
        self._save()
        print("[VectorStore] FAISS index reset")
