"""
Gradio-based web user interface for the multimodal RAG system.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import gradio as gr
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from .agent import AgenticRAG
from .vector_store import VectorStoreManager
from .index import process_and_index_directory
from .config import config


class GradioUI:
    """
    Gradio-based user interface for the RAG system.
    """
    
    def __init__(
        self,
        rag_agent: Optional[AgenticRAG] = None,
        config_instance: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize Gradio UI.
        
        Args:
            rag_agent: RAG agent instance
            config_instance: Configuration instance
            **kwargs: Additional arguments
        """
        self.config = config_instance or config
        self.rag_agent = rag_agent
        self.chat_history = []
        
        # UI components
        self.interface = None
        self._setup_interface()
    
    def _setup_interface(self):
        """Setup Gradio interface."""
        with gr.Blocks(
            title="Property Data RAG System",
            theme=gr.themes.Soft(),
            css="""
            .container {
                max-width: 1200px;
                margin: auto;
            }
            .chat-container {
                height: 500px;
                overflow-y: auto;
            }
            .status-box {
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            """
        ) as interface:
            
            gr.Markdown(
                """
                # 🏠 Property Data RAG System
                
                Upload and analyze property documents using AI-powered search and question answering.
                
                **Features:**
                - 📄 Multi-format document processing (PDF, text files)
                - 🖼️ Image extraction from PDFs
                - 🔍 Intelligent document search
                - 💬 Natural language Q&A
                - ☁️ Powered by AWS Bedrock
                """
            )
            
            with gr.Tab("💬 Chat"):
                self._setup_chat_tab()
            
            with gr.Tab("📚 Document Management"):
                self._setup_document_tab()
            
            with gr.Tab("⚙️ Settings"):
                self._setup_settings_tab()
        
        self.interface = interface
    
    def _setup_chat_tab(self):
        """Setup chat interface tab."""
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False,
                    type="messages"
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about your property documents...",
                        label="Your Question",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            with gr.Column(scale=1):
                status_display = gr.Markdown("### System Status\n*Ready*")
                
                gr.Markdown("### Quick Examples")
                example_questions = [
                    "What is the condition of the roof?",
                    "Are there any structural issues?",
                    "What repairs are recommended?",
                    "Summarize the key findings",
                    "What are the main concerns?"
                ]
                
                for question in example_questions:
                    example_btn = gr.Button(
                        question,
                        variant="outline",
                        size="sm"
                    )
                    example_btn.click(
                        lambda q=question: q,
                        outputs=msg_input
                    )
        
        # Event handlers
        def respond(message, history):
            """Handle chat response."""
            if not message.strip():
                return history, ""
            
            if self.rag_agent is None:
                bot_response = "⚠️ No documents loaded. Please upload documents in the Document Management tab first."
            else:
                try:
                    result = self.rag_agent.run(message)
                    
                    # Handle new response format and check for images
                    if isinstance(result, dict):
                        bot_response = self._format_response(result.get("response", ""))
                        
                        # Check if response contains image references
                        response_text = result.get("response", "")
                        if "image" in message.lower() or "picture" in message.lower():
                            # Search for documents with images
                            image_docs = self._find_image_documents(message)
                            if image_docs:
                                bot_response += self._format_image_response(image_docs)
                        
                        retrieved_images = result.get("images", [])
                        if retrieved_images:
                            bot_response += f"\n\n📊 **Retrieved {len(retrieved_images)} diagram(s)/image(s):**\n"
                            for i, img in enumerate(retrieved_images):
                                page = img.get('page', 'unknown')
                                source = img.get('source', 'unknown')
                                size = img.get('size', 'unknown')
                                
                                # Format size info
                                if isinstance(size, (list, tuple)) and len(size) == 2:
                                    size_str = f"{size[0]}x{size[1]} pixels"
                                else:
                                    size_str = str(size)
                                
                                bot_response += f"• **Image {i+1}:** Page {page} of {Path(str(source)).name} ({size_str})\n"
                    else:
                        # Fallback for old string format
                        bot_response = self._format_response(str(result))
                        
                except Exception as e:
                    logging.error(f"Error in chat response: {e}")
                    bot_response = f"❌ Error: {str(e)}"
            
            # Format for messages API
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_response})
            return history, ""
        
        def clear_chat():
            """Clear chat history."""
            self.chat_history = []
            return []
        
        # Connect events
        send_btn.click(respond, [msg_input, chatbot], [chatbot, msg_input])
        msg_input.submit(respond, [msg_input, chatbot], [chatbot, msg_input])
        clear_btn.click(clear_chat, outputs=chatbot)
    
    def _setup_document_tab(self):
        """Setup document management tab."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📁 Upload Documents")
                
                file_upload = gr.File(
                    label="Upload Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md"],
                    height=150
                )
                
                with gr.Row():
                    process_btn = gr.Button("Process Documents", variant="primary")
                    clear_docs_btn = gr.Button("Clear All Documents", variant="secondary")
                
                processing_status = gr.Markdown("*No documents uploaded*")
                
            with gr.Column():
                gr.Markdown("### 📊 Document Statistics")
                doc_stats = gr.Markdown("*No statistics available*")
                
                gr.Markdown("### 🔍 Search Documents")
                search_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter search terms..."
                )
                search_btn = gr.Button("Search")
                search_results = gr.Markdown("*Enter a search query*")
        
        # Event handlers
        def process_documents(files):
            """Process uploaded documents."""
            if not files:
                return "⚠️ No files selected"
            
            try:
                # Create temporary directory for uploaded files
                import tempfile
                import shutil
                from pathlib import Path
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Copy uploaded files
                    for file in files:
                        if file is not None:
                            src_path = Path(file.name)
                            dst_path = temp_path / src_path.name
                            shutil.copy2(src_path, dst_path)
                    
                    # Process documents
                    vector_store_manager = process_and_index_directory(
                        directory_path=str(temp_path),
                        config_instance=self.config
                    )
                    
                    # Update RAG agent
                    from .bedrock_client import create_bedrock_llm
                    bedrock_config = self.config.get_bedrock_config()
                    llm = create_bedrock_llm(bedrock_config)
                    
                    self.rag_agent = AgenticRAG(
                        vector_store_manager=vector_store_manager,
                        llm=llm,
                        config_instance=self.config
                    )
                    
                    doc_count = vector_store_manager.get_count()
                    return f"✅ Successfully processed {len(files)} files, created {doc_count} document chunks"
                    
            except Exception as e:
                logging.error(f"Error processing documents: {e}")
                return f"❌ Error processing documents: {str(e)}"
        
        def clear_documents():
            """Clear all documents."""
            if self.rag_agent and hasattr(self.rag_agent, 'vector_stores'):
                for store_info in self.rag_agent.vector_stores:
                    store_info['store'].delete_collection()
            
            self.rag_agent = None
            return "🗑️ All documents cleared"
        
        def update_stats():
            """Update document statistics."""
            if self.rag_agent is None:
                return "No documents loaded"
            
            try:
                store_info = self.rag_agent.get_vector_store_info()
                if not store_info:
                    return "No vector stores available"
                
                stats = []
                for store in store_info:
                    stats.append(f"**{store['name']}**: {store['document_count']} documents")
                
                return "\n".join(stats)
                
            except Exception as e:
                return f"Error getting stats: {str(e)}"
        
        def search_documents(query):
            """Search documents."""
            if not query.strip():
                return "Please enter a search query"
            
            if self.rag_agent is None:
                return "No documents loaded"
            
            try:
                # Get documents from first vector store
                if self.rag_agent.vector_stores:
                    store = self.rag_agent.vector_stores[0]['store']
                    docs = store.similarity_search(query, k=3)
                    
                    if not docs:
                        return "No relevant documents found"
                    
                    results = []
                    for i, doc in enumerate(docs, 1):
                        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        source = doc.metadata.get('source', 'Unknown')
                        results.append(f"**Result {i}** (Source: {source})\n{content}")
                    
                    return "\n\n".join(results)
                else:
                    return "No vector stores available"
                    
            except Exception as e:
                return f"Error searching: {str(e)}"
        
        # Connect events
        process_btn.click(process_documents, inputs=file_upload, outputs=processing_status)
        clear_docs_btn.click(clear_documents, outputs=processing_status)
        search_btn.click(search_documents, inputs=search_input, outputs=search_results)
        
        # Auto-update stats periodically
        processing_status.change(lambda: update_stats(), outputs=doc_stats)
    
    def _setup_settings_tab(self):
        """Setup settings tab."""
        with gr.Column():
            gr.Markdown("### ⚙️ System Configuration")
            
            # Model settings
            with gr.Group():
                gr.Markdown("#### 🤖 Model Settings")
                
                model_dropdown = gr.Dropdown(
                    choices=[
                        "anthropic.claude-3-sonnet-20240229-v1:0",
                        "anthropic.claude-3-haiku-20240307-v1:0",
                        "amazon.titan-text-express-v1",
                        "amazon.titan-text-lite-v1"
                    ],
                    value=self.config.get("model", "text_generation"),
                    label="Text Generation Model"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=self.config.get("model", "temperature"),
                    step=0.1,
                    label="Temperature"
                )
                
                max_tokens_slider = gr.Slider(
                    minimum=100,
                    maximum=8000,
                    value=self.config.get("model", "max_tokens"),
                    step=100,
                    label="Max Tokens"
                )
            
            # Retrieval settings
            with gr.Group():
                gr.Markdown("#### 🔍 Retrieval Settings")
                
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=self.config.get("retrieval", "k"),
                    step=1,
                    label="Number of Documents to Retrieve"
                )
                
                score_threshold_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=self.config.get("retrieval", "score_threshold", 0.7),
                    step=0.05,
                    label="Score Threshold"
                )
            
            save_settings_btn = gr.Button("Save Settings", variant="primary")
            settings_status = gr.Markdown("*Settings ready*")
        
        def save_settings(model, temp, max_tokens, k, score_thresh):
            """Save configuration settings."""
            try:
                self.config.set("model", "text_generation", model)
                self.config.set("model", "temperature", temp)
                self.config.set("model", "max_tokens", max_tokens)
                self.config.set("retrieval", "k", k)
                self.config.set("retrieval", "score_threshold", score_thresh)
                
                # Save to file
                self.config.save_config()
                
                return "✅ Settings saved successfully"
                
            except Exception as e:
                return f"❌ Error saving settings: {str(e)}"
        
        save_settings_btn.click(
            save_settings,
            inputs=[model_dropdown, temperature_slider, max_tokens_slider, k_slider, score_threshold_slider],
            outputs=settings_status
        )
    
    def _find_image_documents(self, query: str) -> List[Any]:
        """Find documents that contain images based on query."""
        try:
            if self.rag_agent is None:
                return []
            
            # Search for image documents
            if hasattr(self.rag_agent, 'vector_store_manager'):
                results = self.rag_agent.vector_store_manager.similarity_search(
                    query, k=10
                )
                # Filter for image documents
                image_docs = [doc for doc in results if doc.metadata.get('type') == 'image']
                return image_docs
            
        except Exception as e:
            logging.error(f"Error finding image documents: {e}")
        
        return []
    
    def _format_image_response(self, image_docs: List[Any]) -> str:
        """Format image documents for display."""
        if not image_docs:
            return ""
        
        response = f"\n\n🖼️ **Found {len(image_docs)} Images:**\n\n"
        
        for i, doc in enumerate(image_docs):
            metadata = doc.metadata
            report_id = metadata.get('report_id', 'Unknown')
            page_num = metadata.get('page_number', 'Unknown')
            source_file = metadata.get('source_file', 'Unknown')
            has_raw_data = metadata.get('has_raw_data', False)
            
            response += f"**Image {i+1}:**\n"
            response += f"• Report ID: {report_id}\n"
            response += f"• Source: {source_file}\n"
            if page_num != 'Unknown':
                response += f"• Page: {page_num}\n"
            response += f"• Raw Image Available: {'✅ Yes' if has_raw_data else '❌ No'}\n"
            
            # Show file path if available
            image_file_path = metadata.get('image_file_path')
            image_filename = metadata.get('image_filename')
            if image_file_path:
                response += f"• 📁 File Path: `{image_file_path}`\n"
            if image_filename:
                response += f"• 📷 Filename: `{image_filename}`\n"
            
            response += "\n"
        
        if any(doc.metadata.get('has_raw_data', False) for doc in image_docs):
            response += "💡 **Note**: Images with raw data can be extracted and displayed. In a production system, these would be shown directly in the interface.\n"
        
        return response
    
    def _format_response(self, response: str) -> str:
        """Format response for better display."""
        # Convert XML-like tags to markdown
        response = re.sub(r'<thinking>(.*?)</thinking>', '', response, flags=re.DOTALL)
        response = re.sub(r'<analysis>(.*?)</analysis>', r'**Analysis:**\n\1', response, flags=re.DOTALL)
        response = re.sub(r'<summary>(.*?)</summary>', r'**Summary:**\n\1', response, flags=re.DOTALL)
        
        # Clean up extra whitespace
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = response.strip()
        
        return response
    
    def launch(
        self,
        share: bool = False,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        **kwargs
    ):
        """
        Launch the Gradio interface.
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
            **kwargs: Additional Gradio launch arguments
        """
        if self.interface is None:
            raise ValueError("Interface not initialized")
        
        ui_config = self.config.get("ui") or {}
        
        launch_args = {
            "share": share or ui_config.get("share", False),
            "server_name": server_name,
            "server_port": server_port or ui_config.get("port", 7860),
            "show_error": True,
            **kwargs
        }
        
        logging.info(f"Launching Gradio interface on {server_name}:{server_port}")
        
        try:
            self.interface.launch(**launch_args)
        except Exception as e:
            logging.error(f"Error launching Gradio interface: {e}")
            raise


def create_ui(
    rag_agent: Optional[AgenticRAG] = None,
    config_instance: Optional[Any] = None
) -> GradioUI:
    """
    Create Gradio UI instance.
    
    Args:
        rag_agent: RAG agent instance
        config_instance: Configuration instance
        
    Returns:
        Gradio UI instance
    """
    return GradioUI(rag_agent=rag_agent, config_instance=config_instance)
