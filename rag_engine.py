# rag_engine.py

import chromadb
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)

class EnhancedRAGEngine:
    """
    Enhanced RAG with better chunking, metadata, and hybrid search.
    Uses ChromaDB for vector storage and SentenceTransformers for embeddings.
    """
    
    def __init__(self, collection_name: str = "video_knowledge", 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG engine
        
        Args:
            collection_name: Name for the vector database collection
            embedding_model: Sentence transformer model to use
        """
        # Determine best available device
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
            
        logger.info(f"🚀 Initializing RAG Engine on device: {self.device}")

        # Initialize ChromaDB Client (Ephemeral/In-Memory for this session)
        self.client = chromadb.Client()
        self.collection_name = collection_name
        
        # Reset collection to start fresh
        self._reset_collection()
        
        logger.info(f"📚 Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        
        self.stats = {
            'total_documents': 0,
            'total_queries': 0,
            'last_query_time': 0
        }
        
    def _reset_collection(self):
        """Helper to delete and recreate the collection safely"""
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            # Collection didn't exist, which is fine
            pass
        except Exception as e:
            logger.warning(f"Note during collection reset: {e}")
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Video transcript knowledge base"}
        )

    def ingest_transcript(self, chunks: List[Dict], metadata: Dict = None):
        """
        Ingest transcript chunks with enhanced metadata
        
        Args:
            chunks: List of chunk dicts from utils.chunk_text()
            metadata: Optional metadata about the source (video title, url, etc.)
        """
        if not chunks:
            logger.warning("⚠️ No chunks provided to ingest.")
            return

        logger.info(f"📚 Indexing {len(chunks)} chunks into RAG...")
        
        documents = []
        chunk_metadata = []
        ids = []
        
        for chunk in chunks:
            # Validate chunk structure
            if 'text' not in chunk or 'id' not in chunk:
                continue

            doc_id = f"chunk_{chunk['id']}"
            ids.append(doc_id)
            documents.append(chunk['text'])
            
            # Build metadata for this chunk
            meta = {
                'chunk_id': chunk['id'],
                'start_pos': chunk.get('start_pos', 0),
                'end_pos': chunk.get('end_pos', 0),
                'word_count': chunk.get('word_count', len(chunk['text'].split()))
            }
            
            # Add source metadata if provided
            if metadata:
                meta.update({
                    'source_title': metadata.get('title', 'Unknown'),
                    'source_url': metadata.get('url', ''),
                    'source_duration': metadata.get('duration', 0)
                })
            
            chunk_metadata.append(meta)
        
        # Generate embeddings in batch
        logger.info("🔢 Generating embeddings...")
        try:
            embeddings = self.embedder.encode(documents, show_progress_bar=False).tolist()
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=chunk_metadata,
                ids=ids
            )
            
            self.stats['total_documents'] = len(documents)
            logger.info(f"✓ Indexed {len(documents)} chunks successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings: {e}")

    def search(self, query: str, n_results: int = 3, 
               filter_metadata: Dict = None) -> List[Dict]:
        """
        Search for relevant chunks
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            List of dicts with document, metadata, and distance
        """
        if not query or not query.strip():
            return []

        logger.debug(f"🔍 Searching for: '{query[:50]}...'")
        
        # Generate query embedding
        query_embed = self.embedder.encode([query]).tolist()
        
        # Search with optional filters
        search_kwargs = {
            'query_embeddings': query_embed,
            'n_results': min(n_results, self.collection.count()) # Prevent asking for more than exists
        }
        
        if filter_metadata:
            search_kwargs['where'] = filter_metadata
        
        try:
            results = self.collection.query(**search_kwargs)
        except Exception as e:
            logger.error(f"Search query failed: {e}")
            return []
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0,
                    'id': results['ids'][0][i] if results['ids'] else ''
                })
        
        self.stats['total_queries'] += 1
        logger.debug(f"✓ Found {len(formatted_results)} results")
        
        return formatted_results
    
    def search_with_context(self, query: str, n_results: int = 2,
                           context_window: int = 1) -> str:
        """
        Search and return results with surrounding context (previous/next chunks).
        
        Args:
            query: Search query
            n_results: Number of chunks to retrieve
            context_window: Number of adjacent chunks to include
        
        Returns:
            Formatted string with context
        """
        results = self.search(query, n_results)
        
        if not results:
            return "No relevant information found."
        
        # Build context-aware response
        response_parts = []
        processed_ids = set()
        
        for i, result in enumerate(results, 1):
            main_chunk_id = result['metadata'].get('chunk_id')
            
            # Skip if we've already covered this area via another result's context
            if main_chunk_id in processed_ids:
                continue

            # Gather adjacent chunks based on ID math
            context_chunks = []
            
            # Iterate from -window to +window
            for offset in range(-context_window, context_window + 1):
                target_id = main_chunk_id + offset
                processed_ids.add(target_id)
                
                db_id = f"chunk_{target_id}"
                
                try:
                    # Fetch specific ID
                    adj_result = self.collection.get(ids=[db_id])
                    if adj_result['documents']:
                        context_chunks.append((target_id, adj_result['documents'][0]))
                except Exception:
                    # Chunk might not exist (start/end of video)
                    continue
            
            # Sort by ID to ensure correct reading order
            context_chunks.sort(key=lambda x: x[0])
            
            # Combine text
            combined_text = ' '.join([text for _, text in context_chunks])
            
            response_parts.append(f"[Relevant Excerpt {i}]\n...{combined_text}...\n")
        
        return '\n'.join(response_parts)
    
    def get_related_chunks(self, chunk_id: int, n_related: int = 2) -> List[str]:
        """Get semantically related chunks (not just adjacent)"""
        try:
            # Get the source chunk
            result = self.collection.get(ids=[f"chunk_{chunk_id}"])
            if not result['documents']:
                return []
            
            source_text = result['documents'][0]
            
            # Search for similar content
            # We ask for n+1 because the top result will likely be the chunk itself
            similar = self.search(source_text, n_results=n_related + 1)
            
            # Filter out the source chunk itself based on ID
            related = [r['document'] for r in similar 
                      if r['metadata'].get('chunk_id') != chunk_id]
            
            return related[:n_related]
        except Exception as e:
            logger.warning(f"Could not find related chunks for ID {chunk_id}: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get RAG engine statistics"""
        collection_count = self.collection.count()
        return {
            **self.stats,
            'collection_size': collection_count,
            'device': self.device
        }
    
    def clear(self):
        """Clear all data from collection"""
        self._reset_collection()
        self.stats['total_documents'] = 0
        logger.info(f"Cleared collection: {self.collection_name}")