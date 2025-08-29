"""
Property Retrieval RAG System

A multimodal Retrieval-Augmented Generation system for property document analysis
with diagram and image retrieval capabilities.
"""

__version__ = "1.0.0"
__author__ = "Property Data RAG Team"

# Core modules
from .config import config
from .bedrock_client import create_bedrock_llm, create_bedrock_embeddings
from .vector_store import VectorStoreManager
from .index import process_and_index_directory, DocumentProcessor
from .agent import AgenticRAG
from .ui import create_ui

__all__ = [
    "config",
    "create_bedrock_llm",
    "create_bedrock_embeddings", 
    "VectorStoreManager",
    "process_and_index_directory",
    "DocumentProcessor",
    "AgenticRAG",
    "create_ui"
]