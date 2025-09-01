"""
Vector store operations and management for storing and retrieving document embeddings.
Supports Milvus Lite and FAISS vector stores.
"""
import os
import pickle
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Optional FAISS import (kept for backward compatibility)
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    FAISS = None  # type: ignore

from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters.html import HTMLSemanticPreservingSplitter
from langchain.embeddings.base import Embeddings

# Milvus Lite (pymilvus) client
try:
    from pymilvus import MilvusClient
except Exception:
    MilvusClient = None  # type: ignore


class VectorStoreManager:
    """
    Manages vector store operations with support for multiple backends.
    """
    
    def __init__(
        self,
        store_type: str = "milvus",
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        **kwargs
    ):
        """
        Initialize vector store manager.
        
        Args:
            store_type: Type of vector store ('milvus' or 'faiss')
            collection_name: Name of the collection
            persist_directory: Directory to persist the vector store
            embeddings: Embeddings model to use
            **kwargs: Additional arguments for vector store
        """
        self.store_type = store_type.lower()
        self.collection_name = collection_name
        self.persist_directory = persist_directory or f"./vectorstore_{store_type}"
        # Use absolute path to avoid any readonly mounts or CWD surprises
        self.persist_directory = str(Path(self.persist_directory).resolve())
        self.embeddings = embeddings
        # For FAISS we keep a LangChain vector store instance
        self.vector_store: Optional[Any] = None
        # For Milvus Lite we manage a client directly
        self.milvus_client: Optional[Any] = None
        self.kwargs = kwargs
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        self._setup_vector_store()
    
    def _setup_vector_store(self):
        """Setup the vector store based on type."""
        if self.embeddings is None:
            raise ValueError("Embeddings model is required")
        
        try:
            if self.store_type == "faiss":
                self._setup_faiss()
            elif self.store_type == "milvus":
                self._setup_milvus()
            else:
                raise ValueError(f"Unsupported vector store type: {self.store_type}")
                
            logging.info(f"Vector store ({self.store_type}) initialized successfully")
            
        except Exception as e:
            logging.error(f"Error setting up vector store: {e}")
            raise
    
    def _setup_faiss(self):
        """Setup FAISS vector store."""
        faiss_index_path = os.path.join(self.persist_directory, f"{self.collection_name}.faiss")
        faiss_pkl_path = os.path.join(self.persist_directory, f"{self.collection_name}.pkl")
        
        if os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path):
            # Load existing FAISS index
            try:
                if FAISS is None:
                    raise ImportError("FAISS is not installed")
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    index_name=self.collection_name,
                    allow_dangerous_deserialization=True
                )
                logging.info(f"Loaded existing FAISS index: {self.collection_name}")
            except Exception as e:
                logging.warning(f"Error loading existing FAISS index: {e}")
                self.vector_store = None
        else:
            # Will be created when documents are added
            self.vector_store = None
    
    def _setup_milvus(self):
        """Setup Milvus Lite vector store (local DB via pymilvus MilvusClient)."""
        if MilvusClient is None:
            raise ImportError("pymilvus is required for Milvus Lite. Install with: pip install -U pymilvus")
        
        # Ensure persist directory is writable; fall back if needed
        if not self._ensure_writable_dir(self.persist_directory):
            fallback_dir = self._get_fallback_milvus_dir()
            logging.warning(f"Persist directory not writable: {self.persist_directory}. Falling back to {fallback_dir}")
            self.persist_directory = fallback_dir
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # DB file lives under the persist directory to keep external behavior unchanged
        self.milvus_db_path = os.path.join(self.persist_directory, f"{self.collection_name}.db")
        # Clean up stale lock file if present (single-process app safety)
        try:
            lock_path = os.path.join(self.persist_directory, f".{self.collection_name}.db.lock")
            if os.path.exists(lock_path) and not self._is_process_holding_lock():
                os.remove(lock_path)
                logging.info(f"Removed stale Milvus Lite lock: {lock_path}")
        except Exception as e:
            logging.warning(f"Could not remove Milvus Lite lock file: {e}")
        logging.info(f"Milvus Lite DB path: {self.milvus_db_path}")
        self.milvus_client = MilvusClient(self.milvus_db_path)
        
        # Create the collection if it does not exist
        try:
            has_collection = False
            try:
                has_collection = self.milvus_client.has_collection(collection_name=self.collection_name)
            except Exception:
                # Older client versions may not have has_collection; try describe
                try:
                    self.milvus_client.describe_collection(collection_name=self.collection_name)
                    has_collection = True
                except Exception:
                    has_collection = False
            
            if not has_collection:
                # Infer embedding dimension
                try:
                    test_vec = self.embeddings.embed_query("dimension_probe")
                    dimension = len(test_vec)
                except Exception as e:
                    raise ValueError(f"Unable to infer embedding dimension: {e}")
                
                # Create collection with FLAT index and cosine metric
                self.milvus_client.create_collection(
                    collection_name=self.collection_name,
                    dimension=dimension,
                    auto_id=True,
                    index_params={"index_type": "FLAT", "metric_type": "COSINE"},
                    enable_dynamic_field=True,
                )
                logging.info(f"Created Milvus Lite collection: {self.collection_name} (dim={dimension})")
        except Exception as e:
            logging.error(f"Error setting up Milvus Lite collection: {e}")
            raise
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            **kwargs: Additional arguments for the vector store
            
        Returns:
            List of document IDs
        """
        if not documents:
            logging.warning("No documents provided to add")
            return []
        
        try:
            if self.store_type == "faiss":
                return self._add_documents_faiss(documents, **kwargs)
            elif self.store_type == "milvus":
                return self._add_documents_milvus(documents, **kwargs)
            else:
                raise ValueError(f"Unsupported vector store type: {self.store_type}")
                
        except Exception as e:
            logging.error(f"Error adding documents to vector store: {e}")
            raise
    
    def _add_documents_faiss(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to FAISS vector store."""
        if self.vector_store is None:
            if FAISS is None:
                raise ImportError("FAISS is not installed")
            # Create new FAISS index
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            # Add to existing index
            self.vector_store.add_documents(documents)
        
        # Save the index
        self.save()
        
        # Return dummy IDs (FAISS doesn't return actual IDs)
        return [f"doc_{i}" for i in range(len(documents))]
    
    def _add_documents_milvus(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to Milvus Lite collection."""
        if self.milvus_client is None:
            self._setup_milvus()
        assert self.milvus_client is not None
        
        # Embed all documents
        texts: List[str] = [doc.page_content for doc in documents]
        vectors: List[List[float]] = self.embeddings.embed_documents(texts)
        
        # Prepare data payloads. Use dynamic fields for metadata
        data: List[Dict[str, Any]] = []
        for i, doc in enumerate(documents):
            meta_value: Any
            try:
                # Try to store metadata as a JSON-serializable dict
                json.dumps(doc.metadata)
                meta_value = doc.metadata
            except Exception:
                # Fallback to string
                meta_value = {"_metadata_str": str(doc.metadata)}
            data.append({
                "vector": vectors[i],
                "text": doc.page_content,
                "metadata": meta_value,
            })
        
        # Track rows before insert to verify success
        try:
            pre_count = self._milvus_row_count_safe()
            self.milvus_client.insert(collection_name=self.collection_name, data=data)
            post_count = self._milvus_row_count_safe()
            if post_count < pre_count + len(documents):
                raise RuntimeError("Milvus Lite insert did not persist as expected (possible readonly DB).")
            # Milvus Lite persists automatically
            return [f"doc_{i}" for i in range(len(documents))]
        except Exception as e:
            err_msg = str(e).lower()
            if "readonly" in err_msg or "read-only" in err_msg or "did not persist" in err_msg:
                # Fallback: switch to a guaranteed-writable directory and retry once
                try:
                    fallback_dir = self._get_fallback_milvus_dir()
                    logging.warning(f"Milvus Lite write failed; switching DB to {fallback_dir} and retrying once")
                    # Close client if possible
                    try:
                        if hasattr(self.milvus_client, 'close'):
                            self.milvus_client.close()
                    except Exception:
                        pass
                    self.persist_directory = fallback_dir
                    Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
                    self._setup_milvus()
                    # Retry insert
                    pre_count = self._milvus_row_count_safe()
                    self.milvus_client.insert(collection_name=self.collection_name, data=data)
                    post_count = self._milvus_row_count_safe()
                    if post_count < pre_count + len(documents):
                        raise RuntimeError("Milvus Lite insert still not persisted after fallback.")
                    return [f"doc_{i}" for i in range(len(documents))]
                except Exception as e2:
                    logging.error(f"Milvus Lite insert retry after fallback failed: {e2}")
                    raise
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Query string
            k: Number of results to return
            score_threshold: Minimum score threshold
            **kwargs: Additional search arguments
            
        Returns:
            List of similar documents
        """
        if self.store_type == "milvus":
            return self._similarity_search_milvus(query, k=k, score_threshold=score_threshold)
        
        if self.vector_store is None:
            logging.warning("Vector store is empty")
            return []
        
        try:
            if score_threshold is not None:
                # Use similarity search with score threshold
                docs_and_scores = self.vector_store.similarity_search_with_score(
                    query, k=k, **kwargs
                )
                return [doc for doc, score in docs_and_scores if score >= score_threshold]
            else:
                return self.vector_store.similarity_search(query, k=k, **kwargs)
                
        except Exception as e:
            logging.error(f"Error performing similarity search: {e}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> List[tuple]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Query string
            k: Number of results to return
            **kwargs: Additional search arguments
            
        Returns:
            List of (document, score) tuples
        """
        if self.store_type == "milvus":
            return self._similarity_search_with_score_milvus(query, k=k)
        
        if self.vector_store is None:
            logging.warning("Vector store is empty")
            return []
        
        try:
            return self.vector_store.similarity_search_with_score(query, k=k, **kwargs)
        except Exception as e:
            logging.error(f"Error performing similarity search with score: {e}")
            raise

    def _similarity_search_milvus(self, query: str, k: int = 5, score_threshold: Optional[float] = None) -> List[Document]:
        if self.milvus_client is None:
            self._setup_milvus()
        assert self.milvus_client is not None
        
        try:
            qvec = self.embeddings.embed_query(query)
            res = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[qvec],
                limit=k,
                output_fields=["text", "metadata"],
            )
            hits = res[0] if isinstance(res, list) and res else []
            out: List[Document] = []
            for hit in hits:
                entity = hit.get("entity", hit)
                text = entity.get("text", "")
                meta = entity.get("metadata", {})
                # Score handling (distance similarity varies by metric). Keep behavior consistent with previous code path
                score = hit.get("distance", None)
                if score_threshold is not None and score is not None and score < score_threshold:
                    # Maintain same comparison direction as original (>=). Here we conservatively filter low scores
                    continue
                out.append(Document(page_content=text, metadata=meta if isinstance(meta, dict) else {"metadata": meta}))
            return out
        except Exception as e:
            logging.error(f"Error performing Milvus Lite search: {e}")
            raise

    def _similarity_search_with_score_milvus(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if self.milvus_client is None:
            self._setup_milvus()
        assert self.milvus_client is not None
        
        try:
            qvec = self.embeddings.embed_query(query)
            res = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[qvec],
                limit=k,
                output_fields=["text", "metadata"],
            )
            hits = res[0] if isinstance(res, list) and res else []
            out: List[Tuple[Document, float]] = []
            for hit in hits:
                entity = hit.get("entity", hit)
                text = entity.get("text", "")
                meta = entity.get("metadata", {})
                score = float(hit.get("distance", 0.0))
                out.append((Document(page_content=text, metadata=meta if isinstance(meta, dict) else {"metadata": meta}), score))
            return out
        except Exception as e:
            logging.error(f"Error performing Milvus Lite search with score: {e}")
            raise

    def _milvus_row_count_safe(self) -> int:
        try:
            if self.milvus_client is None:
                return 0
            stats = None
            try:
                stats = self.milvus_client.get_collection_stats(collection_name=self.collection_name)
            except Exception:
                stats = None
            if isinstance(stats, dict) and 'row_count' in stats:
                return int(stats.get('row_count', 0))
            # Fallback: query count
            try:
                rows = self.milvus_client.query(collection_name=self.collection_name, filter="id >= 0", output_fields=["id"])  # type: ignore
                return len(rows) if isinstance(rows, list) else 0
            except Exception:
                return 0
        except Exception:
            return 0

    def _is_process_holding_lock(self) -> bool:
        # Best-effort: in this single-process app, assume no other holder
        return False

    def _ensure_writable_dir(self, path: str) -> bool:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            test_file = os.path.join(path, ".writetest")
            with open(test_file, 'w') as f:
                f.write("ok")
            os.remove(test_file)
            return True
        except Exception:
            return False

    def _get_fallback_milvus_dir(self) -> str:
        # Prefer user-level directory, fall back to /tmp
        try:
            home = str(Path.home())
            lib_dir = os.path.join(home, "Library", "Application Support", "PropertyMeasurementRAG", "milvus")
            if self._ensure_writable_dir(lib_dir):
                return lib_dir
        except Exception:
            pass
        tmp_dir = os.path.join("/tmp", "property_measurement_rag", "milvus")
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        return tmp_dir
    
    def save(self):
        """Save the vector store to disk."""
        if self.vector_store is None:
            logging.warning("No vector store to save")
            return
        
        try:
            if self.store_type == "faiss":
                if self.vector_store is not None:
                    self.vector_store.save_local(
                        self.persist_directory,
                        index_name=self.collection_name
                    )
            elif self.store_type == "milvus":
                # Milvus Lite persists automatically to the DB file
                pass
            logging.info(f"Vector store saved to {self.persist_directory}")
        except Exception as e:
            logging.error(f"Error saving vector store: {e}")
            raise

    # Added public utility methods used elsewhere in the codebase
    def get_count(self) -> int:
        """Return number of vectors/documents in the store."""
        if self.vector_store is None:
            return 0
        try:
            if self.store_type == 'faiss' and self.vector_store is not None:
                return getattr(self.vector_store.index, 'ntotal', 0)
            if self.store_type == 'milvus' and self.milvus_client is not None:
                try:
                    stats = self.milvus_client.get_collection_stats(collection_name=self.collection_name)
                    # Depending on client version, stats may be dict with 'row_count'
                    if isinstance(stats, dict):
                        return int(stats.get('row_count', 0))
                except Exception:
                    # Fallback: try query count
                    try:
                        rows = self.milvus_client.query(collection_name=self.collection_name, filter="id >= 0", output_fields=["id"])  # type: ignore
                        return len(rows) if isinstance(rows, list) else 0
                    except Exception:
                        return 0
            return 0
        except Exception:
            return 0

    def get_info(self) -> Dict[str, Any]:
        """Return basic info about the vector store."""
        return {
            'store_type': self.store_type,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory,
            'count': self.get_count(),
            'initialized': self.vector_store is not None
        }

    def delete_collection(self):
        """Delete the persisted collection files."""
        if self.store_type == 'faiss':
            try:
                faiss_index_path = os.path.join(self.persist_directory, f"{self.collection_name}.faiss")
                faiss_pkl_path = os.path.join(self.persist_directory, f"{self.collection_name}.pkl")
                for p in [faiss_index_path, faiss_pkl_path]:
                    if os.path.exists(p):
                        os.remove(p)
                        logging.info(f"Removed {p}")
            except Exception as e:
                logging.error(f"Error deleting FAISS collection: {e}")
        elif self.store_type == 'milvus':
            try:
                db_path = os.path.join(self.persist_directory, f"{self.collection_name}.db")
                if os.path.exists(db_path):
                    os.remove(db_path)
                    logging.info(f"Removed {db_path}")
                lock_path = os.path.join(self.persist_directory, f".{self.collection_name}.db.lock")
                if os.path.exists(lock_path):
                    os.remove(lock_path)
                    logging.info(f"Removed {lock_path}")
            except Exception as e:
                logging.error(f"Error deleting Milvus Lite DB: {e}")
        self.vector_store = None
        self.milvus_client = None

def create_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter for chunking documents.
    
    Args:
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        Text splitter instance
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

def create_table_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> HTMLSemanticPreservingSplitter:
    """
    Create a HTMLSemanticPreservingSplitter for chunking tables.

    Args:
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        HTMLSemanticPreservingSplitter instance
    """
    
    headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]
    elements_to_preserve = ["table", "ul", "ol"]

    return HTMLSemanticPreservingSplitter(
        headers_to_split_on=headers_to_split_on,
        max_chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        elements_to_preserve=elements_to_preserve,
    )