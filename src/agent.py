"""
Agentic RAG implementation using LangGraph for intelligent document retrieval and generation.
"""
import logging
from typing import Any, Dict, List, Optional, Union, Annotated
from operator import add
from pathlib import Path

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph not available. Install with: pip install langgraph")

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from .config import config
from .vector_store import VectorStoreManager
from .bedrock_client import create_bedrock_llm, create_bedrock_embeddings


class AgentState(BaseModel):
    """State for the RAG agent."""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    documents: List[Document] = Field(default_factory=list)
    query: str = ""
    response: str = ""
    step: str = "start"
    retrieved_images: List[Dict] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class DocumentSearchTool(BaseTool):
    """Tool for searching documents in the vector store."""
    
    name: str = "search_documents"
    description: str = "Search for relevant documents based on a query"
    vector_store_manager: VectorStoreManager = Field(exclude=True)
    k: int = Field(default=5)
    score_threshold: Optional[float] = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Search for documents."""
        try:
            documents = self.vector_store_manager.similarity_search(
                query=query,
                k=self.k,
                score_threshold=self.score_threshold
            )
            
            if not documents:
                return "No relevant documents found."
            
            # Format results
            results = []
            for i, doc in enumerate(documents, 1):
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                metadata = {k: v for k, v in doc.metadata.items() if k not in ['image_data']}  # Exclude image data from display
                results.append(f"Document {i}:\nContent: {content}\nMetadata: {metadata}\n")
            
            return "\n".join(results)
            
        except Exception as e:
            logging.error(f"Error in document search: {e}")
            return f"Error searching documents: {str(e)}"


class AgenticRAG:
    """
    Agentic RAG system using LangGraph for intelligent document retrieval and generation.
    """
    
    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        vector_stores: Optional[List[Dict[str, Any]]] = None,
        llm: Optional[Any] = None,
        config_instance: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize agentic RAG system.
        
        Args:
            vector_store_manager: Single vector store manager (backward compatibility)
            vector_stores: List of vector store configurations
            llm: Language model instance
            config_instance: Configuration instance
            **kwargs: Additional arguments for LLM
        """
        self.config = config_instance or config
        self.llm = llm or self._create_llm(**kwargs)
        self.vector_stores = []
        self.tools = []
        self.graph = None
        
        # Setup vector stores
        if vector_stores:
            self._setup_multiple_vector_stores(vector_stores)
        elif vector_store_manager:
            self._setup_single_vector_store(vector_store_manager)
        else:
            logging.warning("No vector store provided. You'll need to add one before using the agent.")
        
        # Setup agent graph
        if LANGGRAPH_AVAILABLE:
            self._setup_agent_graph()
        else:
            logging.warning("LangGraph not available. Using simple implementation.")
    
    def _create_llm(self, **kwargs):
        """Create LLM instance."""
        bedrock_config = self.config.get_bedrock_config()
        bedrock_config.update(kwargs)
        return create_bedrock_llm(bedrock_config)
    
    def _setup_single_vector_store(self, vector_store_manager: VectorStoreManager):
        """Setup single vector store (backward compatibility)."""
        retrieval_config = self.config.get_retrieval_config()
        
        tool = DocumentSearchTool(
            vector_store_manager=vector_store_manager,
            k=retrieval_config.get("k", 5),
            score_threshold=retrieval_config.get("score_threshold")
        )
        
        self.vector_stores.append({
            'store': vector_store_manager,
            'tool': tool,
            'name': 'search_documents',
            'description': 'Search documents for relevant information.'
        })
        self.tools.append(tool)
    
    def _setup_multiple_vector_stores(self, vector_stores: List[Dict[str, Any]]):
        """Setup multiple vector stores."""
        for i, store_config in enumerate(vector_stores):
            store = store_config['store']
            name = store_config.get('name', f'search_store_{i}')
            description = store_config.get('description', f'Search store {i}')
            k = store_config.get('k', 5)
            score_threshold = store_config.get('score_threshold')
            
            tool = DocumentSearchTool(
                name=name,
                description=description,
                vector_store_manager=store,
                k=k,
                score_threshold=score_threshold
            )
            
            self.vector_stores.append({
                'store': store,
                'tool': tool,
                'name': name,
                'description': description
            })
            self.tools.append(tool)
    
    def _setup_agent_graph(self):
        """Setup LangGraph agent workflow."""
        if not LANGGRAPH_AVAILABLE:
            return
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("search_documents", self._search_documents)
        workflow.add_node("generate_response", self._generate_response)
        
        # Add edges
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "search_documents")
        workflow.add_edge("search_documents", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Compile graph
        self.graph = workflow.compile()
    
    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the user query to determine search strategy."""
        query = state.query
        
        # For now, use the query as-is
        # In a more sophisticated implementation, you could:
        # - Decompose complex queries
        # - Identify key entities
        # - Determine search strategy
        
        state.step = "query_analyzed"
        logging.info(f"Analyzing query: {query}")
        
        return state
    
    def _search_documents(self, state: AgentState) -> AgentState:
        """Search for relevant documents."""
        query = state.query
        all_documents = []
        
        # Search across all vector stores
        for store_info in self.vector_stores:
            try:
                tool = store_info['tool']
                documents = store_info['store'].similarity_search(
                    query=query,
                    k=tool.k,
                    score_threshold=tool.score_threshold
                )
                
                # Add source information
                for doc in documents:
                    doc.metadata['search_tool'] = store_info['name']
                
                all_documents.extend(documents)
                
            except Exception as e:
                logging.error(f"Error searching {store_info['name']}: {e}")
                continue
        
        # Remove duplicates and limit results
        unique_documents = []
        seen_content = set()
        
        for doc in all_documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_documents.append(doc)
        
        # Limit to top results
        max_docs = self.config.get("retrieval", "k", 10)
        state.documents = unique_documents[:max_docs]
        state.step = "documents_retrieved"
        
        logging.info(f"Retrieved {len(state.documents)} relevant documents")
        
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate response based on retrieved documents."""
        query = state.query
        documents = state.documents
        
        if not documents:
            state.response = "I couldn't find any relevant documents to answer your question. Please try rephrasing your query or check if documents have been properly indexed."
            state.step = "completed"
            return state
        
        # Prepare context with enhanced image handling
        context_parts = []
        retrieved_images = []  # Store images for UI display
        
        for i, doc in enumerate(documents, 1):
            content = doc.page_content
            metadata = {k: v for k, v in doc.metadata.items() if k not in ['image_data']}
            
            # Handle image documents with more detail
            if doc.metadata.get('type') == 'image':
                image_info = {
                    'page': metadata.get('page_number', 'unknown'),
                    'source': metadata.get('source', 'unknown source'),
                    'size': metadata.get('image_size', 'unknown'),
                    'index': metadata.get('image_index', i),
                    'image_data': doc.metadata.get('image_data')  # Include for UI
                }
                retrieved_images.append(image_info)
                
                # Enhanced description for LLM
                size_str = f"{image_info['size'][0]}x{image_info['size'][1]}" if isinstance(image_info['size'], (list, tuple)) else str(image_info['size'])
                content = f"[DIAGRAM/IMAGE: Located on page {image_info['page']} of {Path(image_info['source']).name}. Size: {size_str} pixels. This appears to be a visual element that may contain important diagrams, charts, photos, or technical illustrations relevant to the roof report.]"
            
            context_parts.append(f"Document {i}:\n{content}\nSource: {metadata.get('source', 'Unknown')}\n")
        
        # Store retrieved images in state for UI access
        state.retrieved_images = retrieved_images
        
        context = "\n".join(context_parts)
        
        # Create prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context documents. 

Guidelines:
1. Answer questions accurately based on the provided context
2. If information is not in the context, clearly state that
3. Cite relevant documents when making claims
4. For DIAGRAM/IMAGE references, acknowledge them as potentially containing relevant visual information like charts, photos, technical diagrams, or illustrations
5. When diagrams/images are mentioned, suggest that the user should "view the referenced diagrams/images" for visual details
6. Be concise but comprehensive
7. If multiple documents provide different information, synthesize appropriately
8. Pay special attention to visual elements that may contain important technical details, measurements, or visual evidence"""

        user_prompt = f"""Based on the following context documents, please answer this question: {query}

Context Documents:
{context}

Please provide a comprehensive answer based on the available information."""
        
        # Generate response
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                state.response = response.content
            else:
                state.response = str(response)
                
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            state.response = f"I encountered an error while generating the response: {str(e)}"
        
        state.step = "completed"
        return state
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agentic RAG system.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing 'response' (str) and 'images' (list)
        """
        if not self.vector_stores:
            return {"response": "No vector stores available. Please add documents first.", "images": []}
        
        if LANGGRAPH_AVAILABLE and self.graph:
            # Use LangGraph workflow
            initial_state = AgentState(
                query=query,
                messages=[HumanMessage(content=query)]
            )
            
            try:
                final_state = self.graph.invoke(initial_state)
                response = final_state["response"] if isinstance(final_state, dict) else final_state.response
                images = final_state["retrieved_images"] if isinstance(final_state, dict) else getattr(final_state, 'retrieved_images', [])
                return {"response": response, "images": images}
            except Exception as e:
                logging.error(f"Error in LangGraph workflow: {e}")
                # Fallback to simple implementation
                return self._simple_run(query)
        else:
            # Simple implementation without LangGraph
            return self._simple_run(query)
    
    def _simple_run(self, query: str) -> Dict[str, Any]:
        """Simple implementation without LangGraph."""
        try:
            # Search documents
            all_documents = []
            for store_info in self.vector_stores:
                documents = store_info['store'].similarity_search(query, k=3)
                all_documents.extend(documents)
            
            if not all_documents:
                return {"response": "No relevant documents found.", "images": []}
            
            # Process images similar to the LangGraph version
            retrieved_images = []
            context_parts = []
            
            for i, doc in enumerate(all_documents[:5]):
                content = doc.page_content[:300] + "..."
                
                # Handle image documents
                if doc.metadata.get('type') == 'image':
                    image_info = {
                        'page': doc.metadata.get('page_number', 'unknown'),
                        'source': doc.metadata.get('source', 'unknown source'),
                        'size': doc.metadata.get('image_size', 'unknown'),
                        'index': doc.metadata.get('image_index', i),
                        'image_data': doc.metadata.get('image_data')
                    }
                    retrieved_images.append(image_info)
                    
                    # Enhanced description for LLM
                    size_str = f"{image_info['size'][0]}x{image_info['size'][1]}" if isinstance(image_info['size'], (list, tuple)) else str(image_info['size'])
                    content = f"[DIAGRAM/IMAGE: Located on page {image_info['page']} of {Path(image_info['source']).name}. Size: {size_str} pixels. This appears to be a visual element that may contain important diagrams, charts, photos, or technical illustrations relevant to the roof report.]"
                
                context_parts.append(f"{i+1}. {content}")
            
            context = "\n".join(context_parts)
            
            prompt = f"Based on the following context, answer the question: {query}\n\nContext:\n{context}"
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            return {"response": response_text, "images": retrieved_images}
                
        except Exception as e:
            logging.error(f"Error in simple run: {e}")
            return {"response": f"Error processing query: {str(e)}", "images": []}
    
    def add_vector_store(
        self,
        store: VectorStoreManager,
        name: str,
        description: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ):
        """Add a new vector store to the agent."""
        tool = DocumentSearchTool(
            name=name,
            description=description,
            vector_store_manager=store,
            k=k,
            score_threshold=score_threshold
        )
        
        self.vector_stores.append({
            'store': store,
            'tool': tool,
            'name': name,
            'description': description
        })
        self.tools.append(tool)
        
        logging.info(f"Added vector store: {name}")
    
    def remove_vector_store(self, name: str):
        """Remove a vector store by name."""
        self.vector_stores = [vs for vs in self.vector_stores if vs['name'] != name]
        self.tools = [tool for tool in self.tools if getattr(tool, 'name', '') != name]
        
        logging.info(f"Removed vector store: {name}")
    
    def get_vector_store_info(self) -> List[Dict[str, Any]]:
        """Get information about configured vector stores."""
        return [
            {
                'name': vs['name'],
                'description': vs['description'],
                'document_count': vs['store'].get_count(),
                'store_info': vs['store'].get_info()
            }
            for vs in self.vector_stores
        ]
    
    def get_config(self):
        """Get configuration instance."""
        return self.config
    
    @staticmethod
    def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None):
        """Configure logging for this module."""
        logger = logging.getLogger(__name__)
        logger.setLevel(level)
        
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
