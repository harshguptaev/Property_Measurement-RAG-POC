"""
General Query Handler Module for Hierarchical RAG System
Handles ONLY general (non-property-specific) query processing and search across all documents.
This module does NOT contain property-specific query logic or routing functionality.
"""

import json
import logging
import re
from typing import Dict, List, Tuple, Optional
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)


class GeneralQueryHandler:
    """
    Handles ONLY general queries that search across all documents (non-property-specific queries).
    This class does NOT handle property-specific queries or query routing logic.
    """

    def __init__(self, milvus_client: MilvusClient):
        """
        Initialize the GeneralQueryHandler.

        Args:
            milvus_client: Milvus client instance for database operations
        """
        self.milvus_client = milvus_client
        self.embedding_dim = 1536  # Titan embedding dimension

    def is_general_query(self, query: str) -> bool:
        """
        Check if a query is a general query (searches across multiple properties).

        Args:
            query: The search query

        Returns:
            True if query is general, False if property-specific
        """
        # Convert to lowercase for pattern matching
        query_lower = query.lower()

        # Check for general query indicators (queries that span across multiple properties)
        general_indicators = [
            # Questions about multiple properties or comparisons
            r'\b(?:all|every|any|which|what|find|give|show|list)\b.*\b(?:properties?|addresses?|locations?|areas?|pitches?|measurements?)\b',
            r'\b(?:properties?|addresses?|locations?)\b.*\b(?:with|that|having|over|under|greater|less|more|higher|lower)\b',
            r'\b(?:larger|bigger|smaller|greater|less)\b.*\bthan\b',
            r'\b(?:area|pitch|measurement)\b.*\b(?:larger|bigger|smaller|greater|less)\b.*\bthan\b',
            r'\b(?:addresses?|properties?)\b.*\b(?:exceed|above|below|under)\b',
            r'\b(?:find|show|list|get)\b.*\b(?:all|properties?|addresses?)\b',
        ]

        for pattern in general_indicators:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True

        return False

    def _is_area_query(self, query: str) -> bool:
        """
        Check if the query is specifically asking about property areas.

        Args:
            query: The search query

        Returns:
            True if query is about areas, False otherwise
        """
        query_lower = query.lower()
        area_keywords = [
            'area', 'square', 'sq ft', 'sqft', 'square feet', 'square foot',
            'roof area', 'total area', 'building area', 'property area',
            'larger than', 'bigger than', 'greater than', 'more than',
            'smaller than', 'less than', 'under', 'over', 'above', 'below'
        ]

        # Check if query contains area-related terms
        for keyword in area_keywords:
            if keyword in query_lower:
                return True
        return False

    def _search_area_data(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Specialized search for area-related data across all documents.
        Searches for chunks containing area measurements and calculations.

        Args:
            query: Search query about areas
            limit: Number of chunks to retrieve

        Returns:
            List of relevant chunks with area data
        """
        logger.info(f"🔍 Starting specialized area search for: '{query}'")

        try:
            # Create multiple search queries to find area-related chunks
            area_search_terms = [
                "total_roof_area",
                "total area all pitches",
                "roof area measurements",
                "area measurements",
                "total_roof_facets",
                "Measurements - Structure",
                "area diagram",
                "roof measurements"
            ]

            all_results = []

            # Search for each area-related term
            for search_term in area_search_terms:
                try:
                    query_vec = self._get_query_embedding(search_term)

                    res = self.milvus_client.search(
                        collection_name="hierarchical_level2",
                        data=[query_vec],
                        limit=10,  # Get more results per term, we'll filter later
                        output_fields=["chunk_text", "section", "doc_id", "chunk_type", "chunk_id", "source_file"]
                    )

                    if res and res[0]:
                        all_results.extend(res[0])

                except Exception as e:
                    logger.warning(f"Error searching for term '{search_term}': {e}")
                    continue

            # Remove duplicates based on chunk_id
            unique_results = []
            seen_chunk_ids = set()

            for result in all_results:
                chunk_id = result.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    unique_results.append(result)

            # Filter results to only include chunks that actually contain area data
            area_results = []
            for result in unique_results:
                chunk_text = result.get("chunk_text", "").lower()
                if any(term in chunk_text for term in [
                    'total_roof_area', 'total area', 'sq ft', 'square feet',
                    'roof area', 'area:', 'total_roof_facets'
                ]):
                    area_results.append(result)

            # If we still don't have enough area results, fall back to regular search
            if len(area_results) < 5:
                logger.info("Not enough area-specific results, falling back to regular search")
                query_vec = self._get_query_embedding(query)
                fallback_res = self.milvus_client.search(
                    collection_name="hierarchical_level2",
                    data=[query_vec],
                    limit=limit,
                    output_fields=["chunk_text", "section", "doc_id", "chunk_type", "chunk_id", "source_file"]
                )
                if fallback_res and fallback_res[0]:
                    area_results.extend(fallback_res[0])

            # Process results and enrich with document info (limit to requested number)
            final_results = []
            processed_count = 0

            print(f"Fetching Area Data from Level 2 Index (General Search)")

            for result in area_results[:limit * 2]:  # Get more than requested to allow for filtering
                if processed_count >= limit:
                    break

                chunk_info = {
                    "chunk_id": result.get("chunk_id"),
                    "doc_id": result.get("doc_id"),
                    "section": result.get("section"),
                    "chunk_type": result.get("chunk_type"),
                    "chunk_text": result.get("chunk_text"),
                    "distance": result.get("distance", 0),
                    "source_file": result.get("source_file", "unknown.pdf")
                }

                # For general search, we need to get address info from Level 1
                try:
                    # Use a more targeted search for the specific document
                    level1_query = f"document {result.get('doc_id')} address"
                    level1_vec = self._get_query_embedding(level1_query)

                    level1_results = self.milvus_client.search(
                        collection_name="hierarchical_level1",
                        data=[level1_vec],
                        limit=1,
                        filter=f'doc_id == "{result.get("doc_id")}"',
                        output_fields=["address", "summary", "doc_id"]
                    )

                    if level1_results and level1_results[0]:
                        level1_hit = level1_results[0][0]
                        chunk_info["doc_address"] = level1_hit.get("address", "Unknown Address")
                        chunk_info["doc_summary"] = level1_hit.get("summary", "")
                    else:
                        chunk_info["doc_address"] = f"Property {result.get('doc_id')}"
                        chunk_info["doc_summary"] = ""

                except Exception as e:
                    logger.warning(f"Could not retrieve address for doc_id {result.get('doc_id')}: {str(e)}")
                    chunk_info["doc_address"] = f"Property {result.get('doc_id')}"
                    chunk_info["doc_summary"] = ""

                final_results.append(chunk_info)
                processed_count += 1

            # Sort by relevance (distance)
            final_results.sort(key=lambda x: x["distance"])

            self._print_area_search_results(query, final_results[:limit])

            logger.info(f"✅ Area search completed. Found {len(final_results)} relevant chunks")
            return final_results[:limit]

        except Exception as e:
            logger.error(f"Error during area search: {str(e)}")
            return []

    def search_general(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Perform general search directly in Level 2 chunk index (across all documents)

        Args:
            query: Search query
            limit: Number of chunks to retrieve (default: 20 for broader coverage)

        Returns:
            List of relevant chunks with metadata
        """
        logger.info(f"🔍 Starting general search for: '{query}'")

        try:
            # Check if this is an area-related query and modify search strategy
            if self._is_area_query(query):
                return self._search_area_data(query, limit)

            # Generate embedding for query (this would need to be passed or accessed)
            # For now, we'll assume the embedding function is available
            # This will be injected from the main RAG class
            query_vec = self._get_query_embedding(query)

            search_params = {
                "metric_type": "IP",
                "params": {"ef": 64}   # higher ef = better recall, slower search
            }

            # Step 2: Search Level 2 directly (no Level 1 filtering)
            logger.info("📊 Searching Level 2 (Chunk Index) directly...")
            res2 = self.milvus_client.search(
                collection_name="hierarchical_level2",
                data=[query_vec],
                limit=limit,  # Higher limit (20) to retrieve chunks from more properties for general queries
                output_fields=["chunk_text", "section", "doc_id", "chunk_type", "chunk_id", "source_file"]
            )

            if not res2 or not res2[0]:
                logger.warning("No results found in Level 2 Index")
                return []

            # Step 3: Process results and enrich with document info
            final_results = []

            print(f"Fetching Chunks from Level 2 Index (General Search)")

            for result in res2[0]:
                chunk_info = {
                    "chunk_id": result.get("chunk_id"),
                    "doc_id": result.get("doc_id"),
                    "section": result.get("section"),
                    "chunk_type": result.get("chunk_type"),
                    "chunk_text": result.get("chunk_text"),
                    "distance": result.get("distance", 0),
                    "source_file": result.get("source_file", "unknown.pdf")
                }

                # For general search, we need to get address info from Level 1
                # Query Level 1 to get address for this doc_id
                try:
                    level1_results = self.milvus_client.search(
                        collection_name="hierarchical_level1",
                        data=[query_vec],  # Use same query vector
                        limit=1,
                        filter=f'doc_id == "{result.get("doc_id")}"',
                        output_fields=["address", "summary", "doc_id"]
                    )

                    if level1_results and level1_results[0]:
                        level1_hit = level1_results[0][0]
                        chunk_info["doc_address"] = level1_hit.get("address", "Unknown Address")
                        chunk_info["doc_summary"] = level1_hit.get("summary", "")
                    else:
                        chunk_info["doc_address"] = f"Property {result.get('doc_id')}"
                        chunk_info["doc_summary"] = ""

                except Exception as e:
                    logger.warning(f"Could not retrieve address for doc_id {result.get('doc_id')}: {str(e)}")
                    chunk_info["doc_address"] = f"Property {result.get('doc_id')}"
                    chunk_info["doc_summary"] = ""

                final_results.append(chunk_info)

            # Sort by relevance (distance)
            final_results.sort(key=lambda x: x["distance"])

            self._print_general_search_results(query, final_results[:limit])

            logger.info(f"✅ General search completed. Found {len(final_results)} relevant chunks")
            return final_results

        except Exception as e:
            logger.error(f"Error during general search: {str(e)}")
            return []

    def _print_area_search_results(self, query: str, results: List[Dict]) -> None:
        """
        Print area search results in a formatted way, highlighting area measurements.

        Args:
            query: Original search query
            results: Search results from area search
        """
        print(f"\n🔍 Area Search Results for: '{query}'")
        print("=" * 80)

        if not results:
            print("❌ No area data found.")
            return


        for i, result in enumerate(results, 1):
            print(f"--- Area Data Result {i} ---")
            print(f"🏠 Address: {result.get('doc_address', 'N/A')}")
            print(f"📋 Section: {result.get('section', 'N/A')}")
            print(f"🏷️  Type: {result.get('chunk_type', 'N/A')}")
            print(f"🆔 Chunk ID: {result.get('chunk_id', 'N/A')}")

            # Highlight area-related content
            chunk_text = result.get('chunk_text', '')
            print("\n📐 Area Measurements:")
            # Extract and highlight area measurements
            lines = chunk_text.split('\n')
            area_lines = []
            for line in lines:
                line_lower = line.lower()
                if any(term in line_lower for term in [
                    'total_roof_area', 'total area', 'sq ft', 'square feet',
                    'roof area', 'area:', 'total_roof_facets', 'facets'
                ]):
                    area_lines.append(f"  ⭐ {line.strip()}")

            if area_lines:
                print('\n'.join(area_lines))
            else:
                # Fallback to showing full content if no specific area lines found
                content = chunk_text[:400] + "..." if len(chunk_text) > 400 else chunk_text
                print(f"   {content}")

            print()

        print("=" * 80)

    def _print_general_search_results(self, query: str, results: List[Dict]) -> None:
        """
        Print general search results in a formatted way

        Args:
            query: Original search query
            results: Search results from general_search
        """
        print(f"\n🔍 General Search Results for: '{query}'")
        print("=" * 80)

        if not results:
            print("❌ No results found.")
            return


        for i, result in enumerate(results, 1):
            print(f"--- Result {i} ---")
            print(f"🏠 Address: {result.get('doc_address', 'N/A')}")
            print(f"📋 Section: {result.get('section', 'N/A')}")
            print(f"🏷️  Type: {result.get('chunk_type', 'N/A')}")
            print(f"🆔 Chunk ID: {result.get('chunk_id', 'N/A')}")
            print(f"📏 Distance: {result.get('distance', 'N/A'):.4f}")

            # Display chunk content
            chunk_text = result.get('chunk_text', '')
            if len(chunk_text) > 300:
                chunk_text = chunk_text[:300] + "..."
            print(f"📝 Content: {chunk_text}")

            print()

        print("=" * 80)

    def generate_general_response(self, query: str, results: List[Dict]) -> str:
        """
        Generate LLM response for general queries (non-property-specific).

        Args:
            query: The search query
            results: Retrieved chunks from general search

        Returns:
            LLM-generated response string
        """
        if not results:
            return "I couldn't find any relevant information to answer your question."

        # Prepare context from retrieved chunks
        context_parts = []

        for i, result in enumerate(results, 1):
            doc_address = result.get('doc_address', 'Unknown Address')
            section = result.get('section', 'Unknown Section')
            chunk_type = result.get('chunk_type', 'text')
            chunk_text = result.get('chunk_text', '')

            # Clean up chunk text for context
            if chunk_text.startswith('Section:'):
                lines = chunk_text.split('\n')
                content_lines = []
                for line in lines:
                    if line.startswith('Content:'):
                        content_lines.append(line[8:].strip())
                    elif not line.startswith(('Section:', 'Type:')):
                        content_lines.append(line)
                chunk_text = '\n'.join(content_lines).strip()

            context_parts.append(f"""
{doc_address}
Section: {section} ({chunk_type})
Content: {chunk_text}
""")

        context = "\n".join(context_parts)

        # Check if this is an area comparison query
        query_lower = query.lower()
        is_area_comparison = any(phrase in query_lower for phrase in [
            'larger than', 'bigger than', 'greater than', 'more than',
            'smaller than', 'less than', 'under', 'over', 'above', 'below'
        ]) and any(term in query_lower for term in ['area', 'square', 'sq ft', 'sqft'])

        if is_area_comparison:
            prompt = f"""You are a professional EagleView assistant specializing in roofing analysis and property information.

Your task is to analyze property area data and provide specific comparisons based on the retrieved information.

AREA ANALYSIS INSTRUCTIONS:
1. Extract ALL roof area measurements from each property in the chunks
2. Look for "total_roof_area", "total area all pitches", "sq ft", "square feet" values
3. Convert units if necessary (1 sq meter = 10.764 sq ft)
4. Compare each property's area to the specified threshold in the question
5. LIST ONLY the properties that meet the criteria (larger/smaller than threshold)
6. Include the exact area measurement for each qualifying property
7. Format as: "Property Address: Area measurement"
8. If no properties meet the criteria, clearly state this

QUESTION: {query}

RETRIEVED PROPERTY DATA:
{context}

Provide a focused answer listing only the properties that meet the area criteria with their measurements."""
        else:
            prompt = f"""You are a professional EagleView assistant specializing in roofing analysis and property information.

Your task is to provide accurate, relevant information to customer questions based on retrieved property data from multiple properties.

GENERAL QUERY INSTRUCTIONS:
1. Answer questions that span across multiple properties or ask for comparisons
2. Extract and synthesize information from all retrieved chunks
3. Provide comprehensive answers covering relevant properties
4. Include specific measurements, addresses, and technical details
5. Use professional, clear language appropriate for roofing industry customers

QUESTION: {query}

RETRIEVED INFORMATION FROM MULTIPLE PROPERTIES:
{context}

Provide a comprehensive answer based on the data from all relevant properties."""

        try:
            # This should be injected from the main RAG class
            # For now, we'll assume the Bedrock client is available
            if not hasattr(self, '_bedrock_client'):
                raise NotImplementedError("Bedrock client must be injected from main RAG class")

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            response = self._bedrock_client.invoke_model(
                modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text'].strip()

        except Exception as e:
            logger.error(f"Error generating general LLM response: {str(e)}")
            # Fallback to simple summary
            return f"Based on the roofing reports, here's what I found for your query: '{query}'\n\nRetrieved {len(results)} relevant chunks with area information."

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for query. This should be implemented by the main RAG class.

        Args:
            query: Query string

        Returns:
            Embedding vector
        """
        # This is a placeholder - the actual embedding logic should be in the main RAG class
        # We'll need to pass the embedding function or inject it
        raise NotImplementedError("Embedding function must be provided by the main RAG class")

