import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEngine:
    def __init__(self, docs_folder):
        self.docs_folder = docs_folder
        self.documents = []
        self.doc_names = []
        self.vectorizer = TfidfVectorizer(stop_words="english")

        self._load_documents()
        self._build_index()

    def _load_documents(self):
        for filename in os.listdir(self.docs_folder):
            if filename.endswith(".txt"):
                path = os.path.join(self.docs_folder, filename)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    self.documents.append(text)
                    self.doc_names.append(filename)

    def _build_index(self):
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)

    def retrieve(self, query, top_k=1):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors)[0]

        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "doc_name": self.doc_names[idx],
                "content": self.documents[idx]
            })

        return results
