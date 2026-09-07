import os
import logging
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ImportError:
    chromadb = None
    embedding_functions = None
    HAS_CHROMADB = False

from app.config import settings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages persistent vector storage and semantic search operations using ChromaDB.
    Supports similarity checking for news deduplication and semantic document retrieval.
    Includes in-memory fallback if ChromaDB is unavailable.
    """
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.in_memory_store: Dict[str, Dict[str, Any]] = {}
        
        if not HAS_CHROMADB:
            logger.warning("ChromaDB is not installed. Operating in fallback in-memory mode.")
            self.client = None
            self.collection = None
            return

        # Ensure persistence directory exists
        os.makedirs(self.persist_dir, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.ef = self._get_embedding_function()
            try:
                self.collection = self.client.get_or_create_collection(
                    name="news_articles",
                    embedding_function=self.ef,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as coll_err:
                if "embedding function" in str(coll_err).lower() or "conflict" in str(coll_err).lower():
                    logger.warning("Embedding function mismatch detected. Recreating 'news_articles' collection...")
                    try:
                        self.client.delete_collection("news_articles")
                    except Exception:
                        pass
                    self.collection = self.client.get_or_create_collection(
                        name="news_articles",
                        embedding_function=self.ef,
                        metadata={"hnsw:space": "cosine"}
                    )
                else:
                    raise coll_err
            logger.info(f"VectorStoreManager initialized collection 'news_articles' at {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection at {self.persist_dir}: {str(e)}")
            self.client = None
            self.collection = None

    def _get_embedding_function(self):
        """
        Configures embedding function. Uses DefaultEmbeddingFunction (local ONNX/sentence-transformers)
        for reliable free local embedding generation without API quota limits.
        """
        if not HAS_CHROMADB or embedding_functions is None:
            return None

        try:
            logger.info("Using DefaultEmbeddingFunction (local free embeddings) for ChromaDB")
            return embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            logger.warning(f"Could not load DefaultEmbeddingFunction ({str(e)}).")
            return None

    def is_duplicate(self, text: str, threshold: float = 0.85) -> bool:
        """
        Returns True if a semantically equivalent story already exists in the vector collection.
        """
        if not text or not text.strip():
            return False

        if not HAS_CHROMADB or self.collection is None:
            # Fallback exact/basic string match in in-memory store
            text_lower = text.strip().lower()
            for doc_info in self.in_memory_store.values():
                if doc_info["document"].strip().lower() == text_lower:
                    return True
            return False

        try:
            count = self.collection.count()
            if count == 0:
                return False

            results = self.collection.query(
                query_texts=[text],
                n_results=1
            )

            if not results or "distances" not in results or not results["distances"]:
                return False

            distances = results["distances"][0]
            if not distances:
                return False

            distance = distances[0]
            similarity = 1.0 - distance
            max_allowed_distance = 1.0 - threshold

            is_dup = distance <= max_allowed_distance
            logger.debug(f"Deduplication check: distance={distance:.4f}, similarity={similarity:.4f}, threshold={threshold:.4f} -> is_duplicate={is_dup}")
            return is_dup

        except Exception as e:
            logger.error(f"Error checking vector duplication: {str(e)}")
            return False

    def add_article(self, article_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Embeds and stores an article text and associated metadata into ChromaDB.
        """
        if not article_id or not text or not text.strip():
            logger.warning("Attempted to add article with empty ID or text.")
            return False

        clean_metadata = metadata or {}
        sanitized_metadata = {}
        for k, v in clean_metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized_metadata[k] = v
            else:
                sanitized_metadata[k] = str(v)

        if not HAS_CHROMADB or self.collection is None:
            self.in_memory_store[article_id] = {
                "document": text,
                "metadata": sanitized_metadata
            }
            logger.info(f"Added article '{article_id}' to in-memory fallback vector store.")
            return True

        try:
            self.collection.add(
                ids=[article_id],
                documents=[text],
                metadatas=[sanitized_metadata]
            )
            logger.info(f"Successfully added article vector ID '{article_id}' to ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Failed to add article vector ID '{article_id}': {str(e)}")
            return False

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search over stored news articles.
        """
        if not query or not query.strip():
            return []

        if not HAS_CHROMADB or self.collection is None:
            results = []
            query_lower = query.lower()
            for doc_id, data in self.in_memory_store.items():
                if any(w in data["document"].lower() for w in query_lower.split()):
                    results.append({
                        "id": doc_id,
                        "document": data["document"],
                        "metadata": data["metadata"],
                        "distance": 0.1,
                        "similarity": 0.9
                    })
            return results[:top_k]

        try:
            count = self.collection.count()
            if count == 0:
                return []

            actual_k = min(top_k, count)
            results = self.collection.query(
                query_texts=[query],
                n_results=actual_k
            )

            matched_articles = []
            if results and "ids" in results and results["ids"]:
                ids = results["ids"][0]
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]

                for idx in range(len(ids)):
                    dist = distances[idx] if idx < len(distances) else 1.0
                    matched_articles.append({
                        "id": ids[idx],
                        "document": documents[idx] if idx < len(documents) else "",
                        "metadata": metadatas[idx] if idx < len(metadatas) else {},
                        "distance": dist,
                        "similarity": max(0.0, 1.0 - dist)
                    })

            return matched_articles

        except Exception as e:
            logger.error(f"Failed vector search for query '{query}': {str(e)}")
            return []

    def delete_article(self, article_id: str) -> bool:
        """
        Deletes an article vector entry by its ID.
        """
        if not HAS_CHROMADB or self.collection is None:
            if article_id in self.in_memory_store:
                del self.in_memory_store[article_id]
                return True
            return False

        try:
            self.collection.delete(ids=[article_id])
            logger.info(f"Successfully deleted article vector ID '{article_id}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete article vector ID '{article_id}': {str(e)}")
            return False
