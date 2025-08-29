"""
Vector store operations and management for storing and retrieving document embeddings.
Supports FAISS and Chroma vector stores.
"""
import os
import pickle
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from langchain_community.vectorstores import FAISS, Chroma
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings


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
        **kwargs
    ):
        """
        Initialize vector store manager.
        
        Args:
            store_type: Type of vector store ('faiss' or 'chroma')
            collection_name: Name of the collection
            persist_directory: Directory to persist the vector store
            embeddings: Embeddings model to use
            **kwargs: Additional arguments for vector store
        """
        self.store_type = store_type.lower()
        self.collection_name = collection_name
        self.persist_directory = persist_directory or f"./vectorstore_{store_type}"
        self.embeddings = embeddings
        self.vector_store: Optional[VectorStore] = None
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
            elif self.store_type == "chroma":
                self._setup_chroma()
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
    
    def _setup_chroma(self):
        """Setup Chroma vector store."""
        chroma_path = os.path.join(self.persist_directory, self.collection_name)
        
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=chroma_path,
                **self.kwargs
            )
            logging.info(f"Chroma vector store initialized: {self.collection_name}")
        except Exception as e:
            logging.error(f"Error initializing Chroma: {e}")
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
            elif self.store_type == "chroma":
                return self._add_documents_chroma(documents, **kwargs)
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
    
    def _add_documents_chroma(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to Chroma vector store."""
        return self.vector_store.add_documents(documents, **kwargs)
    
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
        if self.vector_store is None:
            logging.warning("Vector store is empty")
            return []
        
        try:
            return self.vector_store.similarity_search_with_score(query, k=k, **kwargs)
        except Exception as e:
            logging.error(f"Error performing similarity search with score: {e}")
            raise
    
    def as_retriever(self, **kwargs) -> Any:
        """
        Get vector store as a retriever.
        
        Args:
            **kwargs: Arguments for the retriever
            
        Returns:
            Vector store retriever
        """
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized or empty")
        
        return self.vector_store.as_retriever(**kwargs)
    
    def save(self):
        """Save the vector store to disk."""
        if self.vector_store is None:
            logging.warning("No vector store to save")
            return
        
        try:
            if self.store_type == "faiss":
                self.vector_store.save_local(
                    self.persist_directory,
                    index_name=self.collection_name
                )
            elif self.store_type == "chroma":
                # Chroma auto-persists, but we can explicitly persist
                if hasattr(self.vector_store, 'persist'):
                    self.vector_store.persist()
            
            logging.info(f"Vector store saved to {self.persist_directory}")
            
        except Exception as e:
            logging.error(f"Error saving vector store: {e}")
            raise
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            if self.store_type == "faiss":
                # Remove FAISS files
                faiss_index_path = os.path.join(self.persist_directory, f"{self.collection_name}.faiss")
                faiss_pkl_path = os.path.join(self.persist_directory, f"{self.collection_name}.pkl")
                
                if os.path.exists(faiss_index_path):
                    os.remove(faiss_index_path)
                if os.path.exists(faiss_pkl_path):
                    os.remove(faiss_pkl_path)
                    
            elif self.store_type == "chroma":
                if hasattr(self.vector_store, 'delete_collection'):
                    self.vector_store.delete_collection()
                else:
                    # Remove directory
                    import shutil
                    chroma_path = os.path.join(self.persist_directory, self.collection_name)
                    if os.path.exists(chroma_path):
                        shutil.rmtree(chroma_path)
            
            self.vector_store = None
            logging.info(f"Collection {self.collection_name} deleted")
            
        except Exception as e:
            logging.error(f"Error deleting collection: {e}")
            raise
    
    def get_count(self) -> int:
        """Get the number of documents in the vector store."""
        if self.vector_store is None:
            return 0
        
        try:
            if self.store_type == "faiss":
                return self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0
            elif self.store_type == "chroma":
                return self.vector_store._collection.count() if hasattr(self.vector_store, '_collection') else 0
            else:
                return 0
        except Exception as e:
            logging.warning(f"Error getting document count: {e}")
            return 0
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the vector store."""
        return {
            "store_type": self.store_type,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "document_count": self.get_count(),
            "is_initialized": self.vector_store is not None
        }


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
