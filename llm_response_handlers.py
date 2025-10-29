"""
LLM Response Handlers for Hierarchical RAG System

Contains all LLM response generation methods for different query flows.
"""

import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class LLMResponseHandlers:
    """Handles different types of LLM responses for various query flows"""

    def __init__(self, bedrock_client, model_id: str):
        """
        Initialize LLM response handlers

        Args:
            bedrock_client: AWS Bedrock client for LLM calls
            model_id: Model ID for Bedrock API calls
        """
        self.bedrock_client = bedrock_client
        self.model_id = model_id

    def _create_fallback_response(self, query: str, results: List[Dict]) -> str:
        """
        Create a fallback response if LLM fails

        Args:
            query: Original user query
            results: Retrieved chunks

        Returns:
            Simple formatted response
        """
        response_parts = [f"Based on the roofing reports, here's what I found for your query: '{query}'\n"]

        for i, result in enumerate(results, 1):
            doc_address = result.get('doc_address', 'Unknown Address')
            section = result.get('section', 'Unknown Section')
            chunk_type = result.get('chunk_type', 'text')

            response_parts.append(f"{i}. Property: {doc_address}")
            response_parts.append(f"   Section: {section} ({chunk_type})")

            # Extract key information from chunk text
            chunk_text = result.get('chunk_text', '')
            if 'area:' in chunk_text.lower():
                # Extract area information
                lines = chunk_text.split('\n')
                for line in lines:
                    if 'area:' in line.lower():
                        response_parts.append(f"   {line.strip()}")

            response_parts.append("")

        return "\n".join(response_parts)

    def handle_similar_addresses_case(self, query: str, results: List[Dict]) -> str:
        """
        Handle the case where all search results are similar addresses

        Args:
            query: Original user query
            results: Retrieved chunks (all address-type)

        Returns:
            LLM-generated response for similar addresses
        """
        context = f"I couldn't find an exact match for the property address you specified. However, I found {len(results)} similar properties that might be what you're looking for:\n\n"
        for i, result in enumerate(results, 1):
            address = result.get('doc_address', 'Unknown Address')
            similarity = result.get('data', {}).get('similarity_score', 0)
            report_id = result.get('report_id', 'Unknown')
            context += f"{i}. {address}\n   Report ID: {report_id}\n\n Similarity Score: {similarity:.2f}\n\n"

        context += "Please check if any of these addresses match what you were looking for, or provide more specific address details for a better search."

        prompt = f"""Based on the search results below, provide a helpful response to the user's query about finding property information.

Query: {query}

Search Results:
{context}

Please provide a response that:
1. Acknowledges that the exact address wasn't found
2. Lists the similar addresses found
3. Suggests the user verify if any match their intended property
4. Offers to help with more specific searches
5. Please sort them in highest similarity to lowest similarity
6. Do not include the similarity score and report id in the response, just the addresses 

Response:"""

        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())
            llm_response = response_body['content'][0]['text'].strip()

            return llm_response

        except Exception as e:
            logger.error(f"Error generating LLM response for similar addresses: {str(e)}")
            return f"I found {len(results)} similar property addresses to what you were looking for. Please check the details above and let me know if you'd like me to search for a specific one."

    def generate_flow1_response(self, query: str, results: List[Dict]) -> str:
        """
        Generate LLM response for Flow 1: Property-specific hierarchical search

        Args:
            query: Original user query
            results: Retrieved chunks from property-specific search

        Returns:
            LLM-generated response string focused on specific property details
        """
        if not results:
            return "I couldn't find any relevant information for this specific property."

        # Check if all results are address-type (similar addresses)
        address_results = [result for result in results if result.get('chunk_type') == 'address']
        if len(address_results) == len(results) and address_results:
            return self.handle_similar_addresses_case(query, results)

        # Prepare context from retrieved chunks
        context_parts = []
        images_found = [result for result in results if result.get('chunk_type') == 'image']

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
Property: {doc_address}
Section: {section} ({chunk_type})
Details: {chunk_text}
""")

        # Add image chunks
        image_chunks = [result for result in results if result.get('chunk_type') == 'image']
        if image_chunks:
            context_parts.append("\n\nVISUAL DATA:")
            for img_chunk in image_chunks:
                context_parts.append(f"""
Image: {img_chunk.get('chunk_id', 'N/A')}
Property: {img_chunk.get('doc_address', 'Unknown Address')}
Description: {img_chunk.get('chunk_text', 'No description available')}
""")

        context = "\n".join(context_parts)

        prompt = f"""You are a professional EagleView assistant specializing in roofing analysis and property information.

Your task is to provide accurate, relevant information to customer questions based on retrieved property data.

INFORMATION PROVIDED:
- Level 1 chunks: High-level property summaries including basic property details, overall roof assessment, structural overview, and general condition reports
- Level 2 chunks: Detailed technical specifications including precise roof measurements (square footage, dimensions), roof geometry details (pitch angles in degrees, facet counts, azimuth directions), structural components (rafter spacing, penetrations), material specifications, and comprehensive measurement data


INSTRUCTIONS:
1. Answer ONLY using the information from the provided chunks
2. Provide complete, accurate measurements and technical details when available
3. If exact information is not available, infer reasonable estimates from related chunk data
4. Be concise but comprehensive - include all relevant measurements and specifications
5. Use professional, clear language appropriate for roofing industry customers
6. Include specific numbers, units, and technical terms as they appear in the chunks
7. Reference image data when relevant to the question
8. You will receive text chunks, image chunks, and table chunks containing comprehensive property data
9. Highlight the most relevant information in the response
QUESTION: {query}

RETRIEVED INFORMATION:
{context}

Provide a clear, professional answer that directly addresses the customer's question with specific details from the data."""

        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())
            llm_text = response_body['content'][0]['text'].strip()

            # Try to parse as JSON
            try:
                json_response = json.loads(llm_text)
                if images_found and 'images_available' not in json_response:
                    json_response['images_available'] = images_found
                return json.dumps(json_response, indent=2)
            except json.JSONDecodeError:
                return llm_text

        except Exception as e:
            logger.error(f"Error generating Flow 1 LLM response: {str(e)}")
            return self._create_fallback_response(query, results)

    def generate_flow2_response(self, query: str, results: List[Dict]) -> str:
        f"""
        Generate LLM response for Flow 2: Non-property-specific criteria search

        Args:
            query: Original user query
            results: Retrieved chunks from criteria-based search across multiple properties

        Returns:
            1. Mention Address importantly imstead of mentioning the property id or any other details
            2. Dont any comapritive analysis only give analysis of the measurement characteristics that are asked in the query
            2. So dont give chunks that are not related to the query if asked to comapare quatities like area, pitch, facets, etc. that is asked
            2. Dont specify the internal document and chunks you are retreiving, just the information you are providing to the user
            3. Dont sound like a robot, sound like a human
            4. Mentioning that you have many properties in database but you are only providing {len(results)} properties to the user
            5. Be clear and concise in your response
        """
        if not results:
            return "I couldn't find any properties matching your criteria."

        # Check if all results are address-type (similar addresses)
        address_results = [result for result in results if result.get('chunk_type') == 'address']
        if len(address_results) == len(results) and address_results:
            return self.handle_similar_addresses_case(query, results)

        # Group results by property for comparison
        property_groups = {}
        images_found = [result for result in results if result.get('chunk_type') == 'image']

        for result in results:
            doc_address = result.get('doc_address', 'Unknown Address')
            if doc_address not in property_groups:
                property_groups[doc_address] = []
            property_groups[doc_address].append(result)

        # Prepare context showing multiple properties
        context_parts = []
        context_parts.append(f"Found {len(property_groups)} properties matching your criteria:\n")

        for i, (address, property_results) in enumerate(property_groups.items(), 1):
            context_parts.append(f"\n--- PROPERTY {i}: {address} ---")

            # Group by section type for cleaner display
            text_chunks = [r for r in property_results if r.get('chunk_type') == 'text']
            image_chunks = [r for r in property_results if r.get('chunk_type') == 'image']

            if text_chunks:
                context_parts.append("Key Specifications:")
                for chunk in text_chunks[:3]:  # Limit to most relevant chunks
                    section = chunk.get('section', 'Unknown')
                    chunk_text = chunk.get('chunk_text', '')

                    # Extract key measurements
                    if 'area' in chunk_text.lower() or 'pitch' in chunk_text.lower() or 'measurement' in chunk_text.lower():
                        # Clean up the text
                        lines = chunk_text.split('\n')
                        key_lines = []
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith(('Section:', 'Type:', 'Content:')):
                                if any(keyword in line.lower() for keyword in ['sq ft', 'pitch', 'degrees', 'facet', 'measurement']):
                                    key_lines.append(line)
                        if key_lines:
                            context_parts.append(f"• {section}: {'; '.join(key_lines[:2])}")

            if image_chunks:
                context_parts.append(f"• Visual data available ({len(image_chunks)} images)")

        # Add image details separately
        if images_found:
            context_parts.append("\n\nVISUAL DATA SUMMARY:")
            for img in images_found[:5]:  # Limit output
                context_parts.append(f"• {img.get('doc_address', 'Unknown')}: {img.get('section', 'Unknown section')}")

        context = "\n".join(context_parts)

        prompt = f"""You are a professional EagleView assistant helping customers compare roofing specifications across multiple properties.

Your task is to analyze and compare roofing data from multiple properties based on the customer's criteria.

SEARCH RESULTS:
Found {len(property_groups)} properties that match your search criteria.

INSTRUCTIONS:
1. Compare and contrast key roofing specifications across properties
2. Highlight similarities and differences in measurements, pitches, areas, etc.
3. Summarize trends or patterns across the matching properties
4. Provide actionable insights for property comparison or selection
5. Include specific measurements and technical details
6. Note when visual data is available for properties
7. Be concise but informative - focus on comparative analysis

QUESTION: {query}

COMPARATIVE PROPERTY DATA:
{context}

Provide a comparative analysis that helps the customer understand roofing specifications across these properties."""

        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2500,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())
            llm_text = response_body['content'][0]['text'].strip()

            # Try to parse as JSON for structured response
            try:
                json_response = json.loads(llm_text)
                if images_found and 'images_available' not in json_response:
                    json_response['images_available'] = images_found
                return json.dumps(json_response, indent=2)
            except json.JSONDecodeError:
                return llm_text

        except Exception as e:
            logger.error(f"Error generating Flow 2 LLM response: {str(e)}")
            return self._create_fallback_response(query, results)

    def generate_llm_response(self, query: str, results: List[Dict]) -> str:
        """
        Use LLM to generate a comprehensive response based on retrieved chunks

        Args:
            query: Original user query
            results: Retrieved chunks from hierarchical search

        Returns:
            LLM-generated response string
        """
        if not results:
            return "I couldn't find any relevant information to answer your question."

        # Prepare context from retrieved chunks
        context_parts = []

        # Check if we have image chunks
        images_found = [result for result in results if result.get('chunk_type') == 'image']

        # Check if all results are address-type (similar addresses)
        address_results = [result for result in results if result.get('chunk_type') == 'address']
        if len(address_results) == len(results) and address_results:
            # All results are similar addresses - delegate to specialized handler
            return self.handle_similar_addresses_case(query, results)

        for i, result in enumerate(results, 1):
            doc_address = result.get('doc_address', 'Unknown Address')
            section = result.get('section', 'Unknown Section')
            chunk_type = result.get('chunk_type', 'text')
            chunk_text = result.get('chunk_text', '')


            # Clean up chunk text for context
            if chunk_text.startswith('Section:'):
                # Remove the redundant section/type prefixes
                lines = chunk_text.split('\n')
                content_lines = []
                for line in lines:
                    if line.startswith('Content:'):
                        content_lines.append(line[8:].strip())  # Remove "Content:" prefix
                    elif not line.startswith(('Section:', 'Type:')):
                        content_lines.append(line)
                chunk_text = '\n'.join(content_lines).strip()

            context_parts.append(f"""
Document {i}: {doc_address}
Section: {section} ({chunk_type})
Content: {chunk_text}
""")

        # Add image chunks directly to context as JSON
        image_chunks = [result for result in results if result.get('chunk_type') == 'image']
        if image_chunks:
            context_parts.append(f"\n\nIMAGE CHUNKS FROM LEVEL 2 SEARCH:")
            for img_chunk in image_chunks:
                context_parts.append(f"""
Image Chunk ID: {img_chunk.get('chunk_id', 'N/A')}
Document: {img_chunk.get('doc_address', 'Unknown Address')}
Section: {img_chunk.get('section', 'Unknown Section')}
Description: {img_chunk.get('chunk_text', 'No description available')}
""")

        context = "\n".join(context_parts)

        prompt = f"""You are a professional EagleView assistant specializing in roofing analysis and property information.

Your task is to provide accurate, relevant information to customer questions based on retrieved property data.

INFORMATION PROVIDED:
- Level 1 chunks: General property information and overviews
- Level 2 chunks: Specific technical details (roof area, facets, pitch, measurements, etc.)

INSTRUCTIONS:
1. Answer ONLY using the information from the provided chunks
2. Provide complete, accurate measurements and technical details when available
3. If exact information is not available, infer reasonable estimates from related chunk data
4. Be concise but comprehensive - include all relevant measurements and specifications
5. Use professional, clear language appropriate for roofing industry customers
6. Include specific numbers, units, and technical terms as they appear in the chunks
7. Reference image data when relevant to the question
8. You will receive text chunks, image chunks, and table chunks containing comprehensive property data

QUESTION: {query}

RETRIEVED INFORMATION:
{context}

Provide a clear, professional answer that directly addresses the customer's question with specific details from the data."""

        try:
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

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())
            llm_text = response_body['content'][0]['text'].strip()

            # Try to parse as JSON
            try:
                json_response = json.loads(llm_text)
                # Add images_found to the response if not already included
                if images_found and 'images_available' not in json_response:
                    json_response['images_available'] = images_found
                return json.dumps(json_response, indent=2)
            except json.JSONDecodeError:
                # If not valid JSON, return as-is
                logger.warning("LLM response is not valid JSON, returning as text")
                return llm_text

        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            # Fallback to simple summary
            return self._create_fallback_response(query, results)
