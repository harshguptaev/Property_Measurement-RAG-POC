"""
Enhanced context management system for integrating structured data with RAG retrieval.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from langchain.schema import Document


class ContextManager:
    """
    Manages context integration between structured important chunks and vector search results.
    """
    
    def __init__(self, important_chunks_dir: str = "important_chunks"):
        """
        Initialize context manager.
        
        Args:
            important_chunks_dir: Directory containing important chunks JSON files
        """
        self.chunks_dir = Path(important_chunks_dir)
        self.loaded_chunks = {}
        self._load_all_chunks()
    
    def _load_all_chunks(self):
        """Load all important chunks from the directory."""
        if not self.chunks_dir.exists():
            logging.warning(f"Important chunks directory not found: {self.chunks_dir}")
            return
        
        for report_dir in self.chunks_dir.iterdir():
            if report_dir.is_dir() and report_dir.name.startswith("report_"):
                chunks_file = report_dir / "important_chunks.json"
                if chunks_file.exists():
                    try:
                        with open(chunks_file, 'r', encoding='utf-8') as f:
                            chunks = json.load(f)
                        
                        report_id = report_dir.name.replace("report_", "")
                        self.loaded_chunks[report_id] = chunks
                        logging.info(f"Loaded {len(chunks)} chunks for report {report_id}")
                    except Exception as e:
                        logging.error(f"Error loading chunks from {chunks_file}: {e}")
    
    def get_report_context(self, report_id: str) -> Dict[str, Any]:
        """
        Get structured context for a specific report.
        
        Args:
            report_id: Report identifier
            
        Returns:
            Dictionary containing structured report data
        """
        chunks = self.loaded_chunks.get(report_id, [])
        
        context = {
            "report_id": report_id,
            "address": None,
            "prepared_for": None,
            "measurements": {},
            "structured_data": {}
        }
        
        for chunk in chunks:
            chunk_type = chunk.get("type")
            chunk_data = chunk.get("data", {})
            
            if chunk_type == "address":
                context["address"] = {
                    "full_address": chunk_data.get("single_line"),
                    "lines": chunk_data.get("lines", [])
                }
            elif chunk_type == "prepared_for":
                context["prepared_for"] = {
                    "name": chunk_data.get("name"),
                    "address": chunk_data.get("single_line"),
                    "phone": chunk_data.get("phone")
                }
            elif chunk_type == "lengths":
                context["measurements"] = chunk_data
            
            context["structured_data"][chunk_type] = chunk_data
        
        return context
    
    def enrich_documents_with_context(self, documents: List[Document], query: str = "") -> List[Document]:
        """
        Enrich retrieved documents with relevant structured context.
        
        Args:
            documents: List of retrieved documents
            query: Original user query for context-aware enrichment
            
        Returns:
            List of enriched documents
        """
        enriched_docs = []
        report_contexts = {}
        
        # Group documents by report ID and collect contexts
        for doc in documents:
            report_id = doc.metadata.get("report_id")
            if report_id and report_id not in report_contexts:
                report_contexts[report_id] = self.get_report_context(report_id)
        
        # Enrich each document
        for doc in documents:
            enriched_doc = Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy()
            )
            
            report_id = doc.metadata.get("report_id")
            if report_id and report_id in report_contexts:
                context = report_contexts[report_id]
                
                # Add structured context to metadata
                enriched_doc.metadata["structured_context"] = context
                
                # Enhance page content with relevant context
                context_additions = []
                
                # Add address context if relevant
                if context.get("address") and any(term in query.lower() for term in ["address", "property", "location"]):
                    context_additions.append(f"Property Address: {context['address']['full_address']}")
                
                # Add measurement context if relevant
                if context.get("measurements") and any(term in query.lower() for term in ["measurement", "area", "length", "pitch", "roof"]):
                    measurements = context["measurements"]
                    key_measurements = []
                    if measurements.get("total_area_all_pitches"):
                        key_measurements.append(f"Total Area: {measurements['total_area_all_pitches']}")
                    if measurements.get("predominant_pitch"):
                        key_measurements.append(f"Predominant Pitch: {measurements['predominant_pitch']}")
                    if key_measurements:
                        context_additions.append("Key Measurements: " + ", ".join(key_measurements))
                
                # Add prepared for context if relevant
                if context.get("prepared_for") and any(term in query.lower() for term in ["prepared", "client", "contact"]):
                    prep_info = context["prepared_for"]
                    if prep_info.get("name"):
                        context_additions.append(f"Prepared For: {prep_info['name']}")
                
                # Append context to page content
                if context_additions:
                    enriched_doc.page_content += "\n\nStructured Context:\n" + "\n".join(context_additions)
            
            enriched_docs.append(enriched_doc)
        
        return enriched_docs
    
    def create_context_summary(self, report_ids: List[str], query: str = "") -> str:
        """
        Create a comprehensive context summary for multiple reports.
        
        Args:
            report_ids: List of report IDs to summarize
            query: User query to focus the summary
            
        Returns:
            Formatted context summary string
        """
        if not report_ids:
            return ""
        
        summary_parts = []
        
        for report_id in report_ids:
            context = self.get_report_context(report_id)
            if not context:
                continue
            
            report_summary = [f"Report {report_id}:"]
            
            # Address information
            if context.get("address"):
                report_summary.append(f"  Property: {context['address']['full_address']}")
            
            # Key measurements
            if context.get("measurements"):
                measurements = context["measurements"]
                if measurements.get("total_area_all_pitches"):
                    report_summary.append(f"  Total Area: {measurements['total_area_all_pitches']}")
                if measurements.get("predominant_pitch"):
                    report_summary.append(f"  Predominant Pitch: {measurements['predominant_pitch']}")
                if measurements.get("ridges"):
                    report_summary.append(f"  Ridges: {measurements['ridges']}")
                if measurements.get("valleys"):
                    report_summary.append(f"  Valleys: {measurements['valleys']}")
            
            # Client information
            if context.get("prepared_for"):
                prep_info = context["prepared_for"]
                if prep_info.get("name"):
                    report_summary.append(f"  Client: {prep_info['name']}")
            
            summary_parts.append("\n".join(report_summary))
        
        return "\n\n".join(summary_parts)
    
    def get_measurement_comparison(self, report_ids: List[str]) -> Dict[str, Any]:
        """
        Compare measurements across multiple reports.
        
        Args:
            report_ids: List of report IDs to compare
            
        Returns:
            Dictionary containing measurement comparisons
        """
        comparison = {
            "reports": [],
            "measurements": {
                "total_area": [],
                "predominant_pitch": [],
                "ridges": [],
                "valleys": [],
                "eaves": [],
                "rakes": []
            }
        }
        
        for report_id in report_ids:
            context = self.get_report_context(report_id)
            if not context.get("measurements"):
                continue
            
            measurements = context["measurements"]
            report_data = {
                "report_id": report_id,
                "address": context.get("address", {}).get("full_address", "Unknown"),
                "measurements": measurements
            }
            comparison["reports"].append(report_data)
            
            # Extract key measurements for comparison
            for key in comparison["measurements"]:
                value = measurements.get(f"total_area_all_pitches" if key == "total_area" else key)
                if value:
                    comparison["measurements"][key].append({
                        "report_id": report_id,
                        "value": value
                    })
        
        return comparison
    
    def search_structured_data(self, query: str, report_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search through structured data using simple text matching.
        
        Args:
            query: Search query
            report_ids: Optional list of report IDs to limit search
            
        Returns:
            List of matching structured data entries
        """
        query_lower = query.lower()
        results = []
        
        search_reports = report_ids if report_ids else list(self.loaded_chunks.keys())
        
        for report_id in search_reports:
            chunks = self.loaded_chunks.get(report_id, [])
            
            for chunk in chunks:
                chunk_type = chunk.get("type")
                chunk_data = chunk.get("data", {})
                
                # Convert chunk data to searchable text
                searchable_text = json.dumps(chunk_data, default=str).lower()
                
                if query_lower in searchable_text or query_lower in chunk_type:
                    results.append({
                        "report_id": report_id,
                        "type": chunk_type,
                        "data": chunk_data,
                        "match_score": searchable_text.count(query_lower)
                    })
        
        # Sort by match score
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results
    
    def get_all_report_ids(self) -> List[str]:
        """Get all available report IDs."""
        return list(self.loaded_chunks.keys())
    
    def reload_chunks(self):
        """Reload all chunks from disk."""
        self.loaded_chunks.clear()
        self._load_all_chunks()
