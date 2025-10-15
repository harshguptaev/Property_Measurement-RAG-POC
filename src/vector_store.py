"""
Vector store operations and management for storing and retrieving document embeddings.
Supports FAISS, Chroma, and Milvus vector stores.
"""
import os
import pickle
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import uuid

from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters.html import HTMLSemanticPreservingSplitter
from langchain.embeddings.base import Embeddings

try:
    from pymilvus import MilvusClient
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    MilvusClient = None


class VectorStoreManager:
    """
    Manages vector store operations with support for multiple backends.
    """
    
    def __init__(
        self,
        store_type: str = "faiss",
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        milvus_uri: Optional[str] = None,
        embedding_dim: int = 1024,  # Default for Titan embeddings
        **kwargs
    ):
        """
        Initialize vector store manager.
        
        Args:
            store_type: Type of vector store ('faiss', 'chroma', or 'milvus')
            collection_name: Name of the collection
            persist_directory: Directory to persist the vector store
            embeddings: Embeddings model to use
            milvus_uri: URI for Milvus connection (for milvus store_type)
            embedding_dim: Dimension of embeddings (required for Milvus)
            **kwargs: Additional arguments for vector store
        """
        self.store_type = store_type.lower()
        self.collection_name = collection_name
        self.persist_directory = persist_directory or f"./vectorstore_{store_type}"
        self.embeddings = embeddings
        self.vector_store: Optional[VectorStore] = None
        self.kwargs = kwargs
        
        # Milvus-specific properties
        self.milvus_uri = milvus_uri or "http://localhost:19530"
        self.embedding_dim = embedding_dim
        self.milvus_client = None
        
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
        """Setup Milvus vector store."""
        if not MILVUS_AVAILABLE:
            raise ValueError("pymilvus is not installed. Install it with: pip install pymilvus")
        
        try:
            self.milvus_client = MilvusClient(uri=self.milvus_uri)
            
            # Check if collection exists
            if self.milvus_client.has_collection(self.collection_name):
                logging.info(f"Using existing Milvus collection: {self.collection_name}")
                # Collection exists, we can use it
                self.vector_store = "milvus_initialized"  # Placeholder to indicate initialized
            else:
                # Collection will be created when documents are added
                logging.info(f"Milvus collection {self.collection_name} will be created when documents are added")
                self.vector_store = None
                
        except Exception as e:
            logging.error(f"Error connecting to Milvus: {e}")
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
        """Add documents to Milvus vector store."""
        if self.milvus_client is None:
            raise ValueError("Milvus client not initialized")
        
        # Create collection if it doesn't exist
        if not self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                dimension=self.embedding_dim,
                metric_type="COSINE",
                consistency_level="Bounded",
            )
            logging.info(f"Created Milvus collection: {self.collection_name}")
        
        # Prepare data for insertion
        records = []
        doc_ids = []
        
        # Get current collection count to generate integer IDs
        try:
            if self.milvus_client.has_collection(self.collection_name):
                collection_stats = self.milvus_client.get_collection_stats(self.collection_name)
                start_id = collection_stats.get('row_count', 0)
            else:
                start_id = 0
        except:
            start_id = 0
        
        for i, doc in enumerate(documents):
            # Generate embeddings for the document
            embedding = self.embeddings.embed_query(doc.page_content)
            if len(embedding) != self.embedding_dim:
                logging.warning(
                    f"Embedding dimension mismatch: got {len(embedding)}, expected {self.embedding_dim}"
                )
                # Try to adjust the embedding dimension
                if len(embedding) > self.embedding_dim:
                    embedding = embedding[:self.embedding_dim]
                else:
                    # Pad with zeros if too short
                    embedding = embedding + [0.0] * (self.embedding_dim - len(embedding))
            
            doc_id = start_id + i + 1  # Use integer ID
            doc_ids.append(str(doc_id))  # Return string for compatibility
            
            record = {
                "id": doc_id,  # Now using integer ID
                "vector": embedding,
                "text": doc.page_content,
                "metadata": str(doc.metadata)  # Convert metadata to string for storage
            }
            records.append(record)
        
        # Insert documents into Milvus
        self.milvus_client.insert(collection_name=self.collection_name, data=records)
        logging.info(f"Added {len(records)} documents to Milvus collection: {self.collection_name}")
        
        # Update vector_store to indicate it's initialized
        self.vector_store = "milvus_initialized"
        
        return doc_ids
    
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
            return self._similarity_search_milvus(query, k, score_threshold, **kwargs)
        
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
            return self._similarity_search_with_score_milvus(query, k, **kwargs)
        
        if self.vector_store is None:
            logging.warning("Vector store is empty")
            return []
        
        try:
            return self.vector_store.similarity_search_with_score(query, k=k, **kwargs)
        except Exception as e:
            logging.error(f"Error performing similarity search with score: {e}")
            raise
    
    def _similarity_search_milvus(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> List[Document]:
        """Perform similarity search using Milvus."""
        if self.milvus_client is None or not self.milvus_client.has_collection(self.collection_name):
            logging.warning("Milvus collection not available")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Perform search
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=k,
                output_fields=["text", "metadata"],
                **kwargs
            )
            
            documents = []
            if results and results[0]:
                for hit in results[0]:
                    # Apply score threshold if specified
                    if score_threshold is not None and hit.get("distance", 0) < score_threshold:
                        continue
                    
                    # Parse metadata back from string
                    metadata_str = hit.get("metadata", "{}")
                    try:
                        metadata = eval(metadata_str) if metadata_str != "{}" else {}
                    except:
                        metadata = {}
                    
                    doc = Document(
                        page_content=hit.get("text", ""),
                        metadata=metadata
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logging.error(f"Error performing Milvus similarity search: {e}")
            return []
    
    def _similarity_search_with_score_milvus(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> List[tuple]:
        """Perform similarity search with scores using Milvus."""
        if self.milvus_client is None or not self.milvus_client.has_collection(self.collection_name):
            logging.warning("Milvus collection not available")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Perform search
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=k,
                output_fields=["text", "metadata"],
                **kwargs
            )
            
            docs_and_scores = []
            if results and results[0]:
                for hit in results[0]:
                    # Parse metadata back from string
                    metadata_str = hit.get("metadata", "{}")
                    try:
                        metadata = eval(metadata_str) if metadata_str != "{}" else {}
                    except:
                        metadata = {}
                    
                    doc = Document(
                        page_content=hit.get("text", ""),
                        metadata=metadata
                    )
                    score = hit.get("distance", 0.0)
                    docs_and_scores.append((doc, score))
            
            return docs_and_scores
            
        except Exception as e:
            logging.error(f"Error performing Milvus similarity search with score: {e}")
            return []
    
    def save(self):
        """Save the vector store to disk."""
        if self.store_type == "milvus":
            # Milvus automatically persists data, no explicit save needed
            logging.info("Milvus data is automatically persisted")
            return
        
        if self.vector_store is None:
            logging.warning("No vector store to save")
            return
        
        try:
            if self.store_type == "faiss":
                self.vector_store.save_local(
                    self.persist_directory,
                    index_name=self.collection_name
                )
            
            logging.info(f"Vector store saved to {self.persist_directory}")
            
        except Exception as e:
            logging.error(f"Error saving vector store: {e}")
            raise

    # Added public utility methods used elsewhere in the codebase
    def get_count(self) -> int:
        """Return number of vectors/documents in the store."""
        if self.store_type == "milvus":
            if self.milvus_client is None or not self.milvus_client.has_collection(self.collection_name):
                return 0
            try:
                stats = self.milvus_client.get_collection_stats(self.collection_name)
                return stats.get("row_count", 0)
            except Exception as e:
                logging.warning(f"Error getting Milvus collection stats: {e}")
                return 0
        
        if self.vector_store is None:
            return 0
        try:
            # FAISS specific
            return getattr(self.vector_store.index, 'ntotal', 0)
        except Exception:
            return 0

    def get_info(self) -> Dict[str, Any]:
        """Return basic info about the vector store."""
        return {
            'store_type': self.store_type,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory,
            'count': self.get_count(),
            'initialized': (
                self.vector_store is not None if self.store_type == "faiss"
                else self.milvus_client is not None if self.store_type == "milvus"
                else False
            )
        }

    def delete_collection(self):
        """Delete the persisted collection."""
        if self.store_type == 'milvus':
            try:
                if self.milvus_client and self.milvus_client.has_collection(self.collection_name):
                    self.milvus_client.drop_collection(self.collection_name)
                    logging.info(f"Dropped Milvus collection: {self.collection_name}")
                self.vector_store = None
            except Exception as e:
                logging.error(f"Error deleting Milvus collection: {e}")
        elif self.store_type == 'faiss':
            try:
                faiss_index_path = os.path.join(self.persist_directory, f"{self.collection_name}.faiss")
                faiss_pkl_path = os.path.join(self.persist_directory, f"{self.collection_name}.pkl")
                for p in [faiss_index_path, faiss_pkl_path]:
                    if os.path.exists(p):
                        os.remove(p)
                        logging.info(f"Removed {p}")
                self.vector_store = None
            except Exception as e:
                logging.error(f"Error deleting FAISS collection: {e}")
        else:
            logging.warning(f"Delete operation not implemented for store type: {self.store_type}")

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