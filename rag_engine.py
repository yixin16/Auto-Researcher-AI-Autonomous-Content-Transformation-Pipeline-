# rag_engine.py

import chromadb
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        self.client = chromadb.Client()
        # Re-create collection to flush old data
        try: self.client.delete_collection("video_knowledge")
        except: pass
        self.collection = self.client.create_collection("video_knowledge")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def ingest_transcript(self, chunks):
        logger.info("📚 Indexing video content for RAG...")
        ids = [str(i) for i in range(len(chunks))]
        embeddings = self.embedder.encode(chunks).tolist()
        self.collection.add(documents=chunks, embeddings=embeddings, ids=ids)

    def search(self, query, n_results=2):
        query_embed = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embed, n_results=n_results)
        return results['documents'][0]