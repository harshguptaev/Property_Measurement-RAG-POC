"""
Gradio-based web user interface for the multimodal RAG system.
"""
import logging
import re
import os
import base64
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import gradio as gr
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from PIL import Image
import numpy as np

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
        self.selected_images = []  # Store selected images for chat
        self.extracted_images_path = Path("extracted_images")  # Path to extracted images
        
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
            .image-gallery {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                max-height: 400px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .image-item {
                width: 150px;
                height: 150px;
                object-fit: cover;
                border: 2px solid transparent;
                border-radius: 5px;
                cursor: pointer;
                transition: border-color 0.3s;
            }
            .image-item:hover {
                border-color: #007bff;
            }
            .image-item.selected {
                border-color: #28a745;
                box-shadow: 0 0 10px rgba(40, 167, 69, 0.5);
            }
            .image-upload-area {
                border: 2px dashed #ddd;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                background-color: #f9f9f9;
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
            
            with gr.Tab("�️ Image Gallery"):
                self._setup_image_gallery_tab()
            
            with gr.Tab("�📚 Document Management"):
                self._setup_document_tab()
            
            with gr.Tab("⚙️ Settings"):
                self._setup_settings_tab()
        
        self.interface = interface
    
    def _setup_chat_tab(self):
        """Setup chat interface tab."""
        with gr.Row():
            with gr.Column(scale=3):
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
                
                # Image input section
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Image (Optional)",
                            type="pil",
                            height=200,
                            sources=["upload", "webcam", "clipboard"],
                            interactive=True
                        )
                        clear_image_btn = gr.Button("Clear Image", variant="secondary", size="sm")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🖼️ Selected from Gallery")
                        selected_images_display = gr.Gallery(
                            label="Selected Images",
                            show_label=False,
                            columns=2,
                            rows=2,
                            height=200,
                            object_fit="cover"
                        )
                        clear_selected_btn = gr.Button("Clear Selected", variant="secondary", size="sm")
                
                clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            with gr.Column(scale=1):
                status_display = gr.Markdown("### System Status\n*Ready*")
                
                # Model capability info
                gr.Markdown("""
                ### 🤖 Current Model
                Check **Settings** tab to select:
                - **Claude 3.5 Sonnet v2**: Best for image analysis
                - **Claude 3 Opus**: Most capable reasoning
                - **Claude 3 Haiku**: Fast & economical
                """)
                
                gr.Markdown("### Quick Examples")
                example_questions = [
                    "What is the condition of the roof?",
                    "Are there any structural issues?",
                    "What repairs are recommended?",
                    "Analyze this image for damage",
                    "Compare the roof images",
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
        def respond(message, history, uploaded_image):
            """Handle chat response with image support."""
            if not message.strip() and uploaded_image is None and not self.selected_images:
                return history, ""
            
            # Process images
            image_context = ""
            processed_images = []
            
            # Handle uploaded image
            if uploaded_image is not None:
                try:
                    # Convert PIL image to base64 for context
                    import io
                    buffered = io.BytesIO()
                    uploaded_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    processed_images.append({
                        "type": "uploaded",
                        "data": img_str,
                        "name": "uploaded_image.png"
                    })
                    image_context += "\n\n[User uploaded an image for analysis]"
                except Exception as e:
                    logging.error(f"Error processing uploaded image: {e}")
            
            # Handle selected images from gallery
            if self.selected_images:
                for img_path in self.selected_images:
                    try:
                        # Ensure img_path is a string
                        if isinstance(img_path, dict):
                            # If it's a dict, try to get path from it
                            img_path = img_path.get('path') or img_path.get('name') or str(img_path)
                        
                        img_path = str(img_path)  # Convert to string
                        
                        if os.path.exists(img_path):
                            with open(img_path, "rb") as img_file:
                                img_str = base64.b64encode(img_file.read()).decode()
                                processed_images.append({
                                    "type": "gallery",
                                    "data": img_str,
                                    "name": Path(img_path).name,
                                    "path": img_path
                                })
                        else:
                            logging.warning(f"Image path does not exist: {img_path}")
                    except Exception as e:
                        logging.error(f"Error processing selected image {img_path}: {e}")
                
                if self.selected_images:
                    # Safely get image names
                    image_names = []
                    for p in self.selected_images:
                        try:
                            if isinstance(p, dict):
                                name = p.get('name') or Path(str(p.get('path', ''))).name
                            else:
                                name = Path(str(p)).name
                            image_names.append(name)
                        except:
                            image_names.append(str(p))
                    
                    image_context += f"\n\n[User selected {len(self.selected_images)} image(s) from gallery: {', '.join(image_names)}]"
            
            # Combine message with image context
            full_message = message + image_context if message.strip() else image_context
            
            if self.rag_agent is None:
                bot_response = "⚠️ No documents loaded. Please upload documents in the Document Management tab first."
            else:
                try:
                    # Enhanced query with image information
                    enhanced_query = full_message
                    if processed_images:
                        enhanced_query += f"\n\nNote: This query includes {len(processed_images)} image(s) for analysis."
                    
                    result = self.rag_agent.run(enhanced_query)
                    
                    # Handle response format
                    if isinstance(result, dict):
                        bot_response = self._format_response(result.get("response", ""))
                        
                        # Add image analysis if images were provided
                        if processed_images:
                            bot_response += self._format_image_analysis(processed_images)
                        
                        retrieved_images = result.get("images", [])
                        if retrieved_images:
                            bot_response += f"\n\n📊 **Retrieved {len(retrieved_images)} diagram(s)/image(s):**\n"
                            for i, img in enumerate(retrieved_images):
                                page = img.get('page', 'unknown')
                                source = img.get('source', 'unknown')
                                size = img.get('size', 'unknown')
                                
                                if isinstance(size, (list, tuple)) and len(size) == 2:
                                    size_str = f"{size[0]}x{size[1]} pixels"
                                else:
                                    size_str = str(size)
                                
                                bot_response += f"• **Image {i+1}:** Page {page} of {Path(str(source)).name} ({size_str})\n"
                    else:
                        bot_response = self._format_response(str(result))
                        
                except Exception as e:
                    logging.error(f"Error in chat response: {e}")
                    bot_response = f"❌ Error: {str(e)}"
            
            # Format for messages API
            history.append({"role": "user", "content": full_message})
            history.append({"role": "assistant", "content": bot_response})
            return history, ""
        
        def clear_chat():
            """Clear chat history."""
            self.chat_history = []
            return []
        
        def clear_uploaded_image():
            """Clear uploaded image."""
            return None
        
        def clear_selected_images():
            """Clear selected images from gallery."""
            self.selected_images = []
            return []
        
        def update_selected_images_display():
            """Update display of selected images."""
            if not self.selected_images:
                return []
            
            image_list = []
            for img_path in self.selected_images:
                try:
                    # Handle both string paths and dict objects
                    if isinstance(img_path, dict):
                        path_str = img_path.get('path') or str(img_path)
                    else:
                        path_str = str(img_path)
                    
                    if os.path.exists(path_str):
                        image_list.append(path_str)
                except Exception as e:
                    logging.warning(f"Error processing image path {img_path}: {e}")
            
            return image_list
        
        # Connect events
        send_btn.click(
            respond, 
            [msg_input, chatbot, image_input], 
            [chatbot, msg_input]
        )
        msg_input.submit(
            respond, 
            [msg_input, chatbot, image_input], 
            [chatbot, msg_input]
        )
        clear_btn.click(clear_chat, outputs=chatbot)
        clear_image_btn.click(clear_uploaded_image, outputs=image_input)
        clear_selected_btn.click(clear_selected_images, outputs=selected_images_display)
        
        # Auto-update selected images display
        clear_selected_btn.click(update_selected_images_display, outputs=selected_images_display)
    
    def _setup_image_gallery_tab(self):
        """Setup image gallery tab for viewing and selecting extracted images."""
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Extracted Images")
                gr.Markdown("Browse and select images extracted from property documents. Selected images can be used in chat for analysis.")
                
                # Report selector
                report_selector = gr.Dropdown(
                    label="Select Report",
                    choices=self._get_available_reports(),
                    value=None,
                    interactive=True
                )
                
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")
                
                # Image gallery
                image_gallery = gr.Gallery(
                    label="Available Images",
                    show_label=True,
                    columns=4,
                    rows=3,
                    height=600,
                    object_fit="cover",
                    allow_preview=True,
                    interactive=True
                )
                
                # Selection controls
                with gr.Row():
                    select_all_btn = gr.Button("Select All", variant="outline", size="sm")
                    clear_selection_btn = gr.Button("Clear Selection", variant="outline", size="sm")
                    add_to_chat_btn = gr.Button("Add to Chat", variant="primary", size="sm")
            
            with gr.Column(scale=1):
                # Selected images info
                gr.Markdown("### 📋 Selection Info")
                selection_info = gr.Markdown("*No images selected*")
                
                # Image details
                gr.Markdown("### 🔍 Image Details")
                image_details = gr.Markdown("*Select an image to view details*")
                
                # Bulk operations
                gr.Markdown("### 🔧 Bulk Operations")
                with gr.Column():
                    export_btn = gr.Button("📤 Export Selected", variant="outline")
                    analyze_btn = gr.Button("🔍 Analyze Selected", variant="outline")
        
        # Event handlers
        def load_images_for_report(report_id):
            """Load images for selected report."""
            if not report_id:
                return [], "*No report selected*"
            
            report_path = self.extracted_images_path / report_id
            if not report_path.exists():
                return [], f"*Report {report_id} not found*"
            
            image_files = []
            supported_formats = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
            
            for img_file in report_path.iterdir():
                if img_file.suffix.lower() in supported_formats:
                    image_files.append(str(img_file))
            
            image_files.sort()  # Sort alphabetically
            
            info_text = f"*Found {len(image_files)} images in {report_id}*"
            return image_files, info_text
        
        def refresh_reports():
            """Refresh available reports."""
            reports = self._get_available_reports()
            return gr.Dropdown(choices=reports, value=None)
        
        def select_all_images(gallery_value):
            """Select all visible images."""
            if gallery_value:
                # Ensure all paths are strings
                self.selected_images = [str(path) for path in gallery_value]
                return self._update_selection_info()
            return "*No images to select*"
        
        def clear_all_selection():
            """Clear all selected images."""
            self.selected_images = []
            return "*No images selected*"
        
        def add_selected_to_chat():
            """Add selected images to chat context."""
            if not self.selected_images:
                return "*No images selected to add*"
            
            count = len(self.selected_images)
            return f"*✅ {count} image(s) added to chat context*"
        
        def show_image_details(evt: gr.SelectData):
            """Show details for selected image."""
            if evt.index is not None and evt.value:
                try:
                    img_path = Path(str(evt.value))  # Ensure it's a string path
                    if img_path.exists():
                        # Get image info
                        img = Image.open(img_path)
                        size = img.size
                        mode = img.mode
                        file_size = img_path.stat().st_size
                        
                        details = f"""**Filename:** {img_path.name}
**Report:** {img_path.parent.name}
**Dimensions:** {size[0]} x {size[1]} pixels
**Color Mode:** {mode}
**File Size:** {file_size:,} bytes
**Path:** `{str(img_path)}`"""
                        
                        return details
                    else:
                        return f"*Image file not found: {evt.value}*"
                except Exception as e:
                    logging.error(f"Error loading image details: {e}")
                    return f"*Error loading image details: {e}*"
            
            return "*Select an image to view details*"
        
        def on_gallery_select(evt: gr.SelectData):
            """Handle image selection in gallery."""
            if evt.index is not None and evt.value:
                img_path = str(evt.value)  # Ensure it's a string
                
                if img_path in self.selected_images:
                    # Deselect
                    self.selected_images.remove(img_path)
                else:
                    # Select
                    self.selected_images.append(img_path)
                
                return self._update_selection_info()
            
            return self._update_selection_info()
        
        # Connect events
        report_selector.change(load_images_for_report, inputs=report_selector, outputs=[image_gallery, selection_info])
        refresh_btn.click(refresh_reports, outputs=report_selector)
        select_all_btn.click(select_all_images, inputs=image_gallery, outputs=selection_info)
        clear_selection_btn.click(clear_all_selection, outputs=selection_info)
        add_to_chat_btn.click(add_selected_to_chat, outputs=selection_info)
        image_gallery.select(show_image_details, outputs=image_details)
        image_gallery.select(on_gallery_select, outputs=selection_info)
    
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
                
                gr.Markdown("""
                **Model Capabilities:**
                - **Claude 3.5 Sonnet v2** (Latest): Best image analysis + text reasoning
                - **Claude 3 Opus**: Most capable reasoning, slower but thorough
                - **Claude 3 Sonnet**: Good balance of speed and capability
                - **Claude 3 Haiku**: Fastest and most economical with vision
                - **Titan models**: Text-only, no image analysis
                """)
                
                model_dropdown = gr.Dropdown(
                    choices=[
                        # Vision-capable Claude 3 models (recommended for image analysis)
                        "anthropic.claude-3-5-sonnet-20241022-v2:0",  # Latest Claude 3.5 Sonnet (Best)
                        "anthropic.claude-3-5-sonnet-20240620-v1:0",   # Previous Claude 3.5 Sonnet
                        "anthropic.claude-3-opus-20240229-v1:0",       # Claude 3 Opus (Most capable)
                        "anthropic.claude-3-sonnet-20240229-v1:0",     # Claude 3 Sonnet (Balanced)
                        "anthropic.claude-3-haiku-20240307-v1:0",      # Claude 3 Haiku (Fast & economical)
                        # Text-only models
                        "amazon.titan-text-express-v1",
                        "amazon.titan-text-lite-v1"
                    ],
                    value="anthropic.claude-3-5-sonnet-20241022-v2:0",  # Default to latest vision model
                    label="🤖 AI Model (Vision models support image analysis)"
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
    
    def _get_available_reports(self) -> List[str]:
        """Get list of available report directories."""
        if not self.extracted_images_path.exists():
            return []
        
        reports = []
        for item in self.extracted_images_path.iterdir():
            if item.is_dir():
                reports.append(item.name)
        
        return sorted(reports)
    
    def _update_selection_info(self) -> str:
        """Update selection information display."""
        if not self.selected_images:
            return "*No images selected*"
        
        count = len(self.selected_images)
        
        # Safely get image names
        image_names = []
        for img in self.selected_images[-3:]:  # Show last 3
            try:
                if isinstance(img, dict):
                    name = img.get('name') or Path(str(img.get('path', ''))).name
                else:
                    name = Path(str(img)).name
                image_names.append(name)
            except Exception as e:
                logging.warning(f"Error getting image name: {e}")
                image_names.append(str(img)[:20])  # Fallback to truncated string
        
        info = f"**Selected:** {count} image(s)\n"
        if count <= 3:
            info += f"**Files:** {', '.join(image_names)}"
        else:
            info += f"**Latest:** {', '.join(image_names)}... (+{count-3} more)"
        
        return info
    
    def _format_image_analysis(self, processed_images: List[Dict]) -> str:
        """Format image analysis information."""
        if not processed_images:
            return ""
        
        analysis = f"\n\n🖼️ **Image Analysis Context:**\n"
        
        for i, img_info in enumerate(processed_images):
            img_type = img_info.get("type", "unknown")
            img_name = img_info.get("name", f"image_{i+1}")
            
            if img_type == "uploaded":
                analysis += f"• **Uploaded Image:** {img_name}\n"
            elif img_type == "gallery":
                img_path = img_info.get("path", "")
                report_name = Path(img_path).parent.name if img_path else "unknown"
                analysis += f"• **Gallery Image:** {img_name} (from {report_name})\n"
        
        analysis += "\n💡 **Note:** The AI can analyze these images for damage, measurements, structural issues, and other property-related insights.\n"
        
        return analysis
    
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
