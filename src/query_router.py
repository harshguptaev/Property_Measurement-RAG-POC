"""
Query Router for Property Measurement RAG System

This module analyzes user queries and determines the appropriate search flow:
- Flow 1: Property-specific queries (address given, find specific information)
- Flow 2: Non-property-specific queries (find properties matching criteria)
- Flow 3: Address not found queries (when user searches for address but no exact match or similar addresses found)
"""

import json
import logging
from typing import Dict, Any, Optional
import boto3
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class QueryAnalysis(BaseModel):
    """Structured analysis of a user query."""
    flow: str  # "1", "2", or "3"
    address: Optional[str] = None  # Extracted address if present
    query: str  # Cleaned user intention/query
    complete_query: str  # Full original query
    relevant_sections: list[str]  # List of relevant data sections for the query
    important_imagery: list[str]  # List of specific image types that are most relevant for this query


class QueryRouter:
    """
    Query Router that uses LLM to analyze queries and determine routing strategy.
    """

    def __init__(self, region_name: str = "us-east-1", model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        """
        Initialize the Query Router

        Args:
            region_name: AWS region for Bedrock
            model_id: Bedrock model ID for text analysis
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region_name)
        self.model_id = model_id
        self.region_name = region_name

        logger.info(f"🔧 Initialized Query Router with model: {model_id}")

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a user query and return structured routing information.

        Args:
            query: The user's query string

        Returns:
            QueryAnalysis object with flow, address, query, complete_query, and relevant_sections
        """
        prompt = f"""You are a query analysis expert for a property measurement and roofing analysis system.

Your task is to analyze the user's query and extract the following information:

1. **flow**: Determine if this is:
   - "1": Property-specific query (user mentions a specific address/property and wants information about it)
   - "2": Non-property-specific query (user wants to find properties that match certain criteria, like "find properties with area > 2000 sq ft")
   - "3": Generate new report query (user requests to create/generate a new property report for an address that may not exist in the current database; also includes cases where address not found and suggested addresses don't work)

2. **address**: If the query mentions a specific address, extract it exactly as written. If no address is mentioned, set to null. For flow "3", if the user provides an address that could not be found (including when suggested/similar addresses are not suitable), still return that address here.

3. **query**: The user's actual intent/question, cleaned of address information if present.

4. **complete_query**: The full original query as provided.

5. **relevant_sections**: Based on the query intent, identify which data section PREFIXES would be most relevant. Return section prefixes that will be matched using LIKE queries. Available section prefixes include:
   - "House Measurements": Basic property info (stories, facets, complexity, attic area (Not the actual area it is a estimate), obstructions)
   - "Roof Measurements": Detailed roof measurements (actual area value, pitch, ridges, hips, valleys, rakes, eaves, etc.) - matches all roof measurement sections including per-structure ones
   - "Pitch Breakdown": Roof pitch percentages and areas by pitch type - matches all pitch breakdown sections including per-structure ones
   - "Waste Calculation": Waste factor calculations for different percentages - matches all waste calculation sections including per-structure ones
   - "Diagrams": Technical diagrams (lengths, pitch degrees, pitch on 12, rafters, azimuth, area, roof penetrations) - include for queries about measurements, pitch, area, or technical details
   - "Imagery": Property photos (top view, north/south/east/west sides) - include for visual inspection queries

   Return an array of the most relevant section names based on what information would help answer the query.

6. **important_imagery**: Based on the query intent, identify which SPECIFIC image types would be most valuable to include in the response. Only include images that directly help answer the query. Available image types include:
   - "Cover_Image": Property overview/cover image - useful for general property queries
   - "Top_View": Aerial/top view - useful for roof structure, layout, and general property overview
   - "North_Side": North side view - useful for queries about north-facing aspects
   - "South_Side": South side view - useful for queries about south-facing aspects
   - "East_Side": East side view - useful for queries about east-facing aspects
   - "West_Side": West side view - useful for queries about west-facing aspects
   - "Area": Roof area measurements diagram - useful for area-related queries
   - "Azimuth": Roof direction/azimuth diagram - useful for orientation/direction queries
   - "Pitch_Degrees": Roof pitch in degrees diagram - useful for pitch-related queries
   - "Pitch_on_12": Roof pitch (rise over 12) diagram - useful for pitch-related queries
   - "Rafters": Rafter structure diagram - useful for structural queries
   - "Roof_Penetrations": Roof penetrations diagram - useful for penetration/ventilation queries
   - "Lengths": Length measurements diagram - useful for dimension queries

   Return an array of the most relevant image type names. Return empty array [] if no images are needed for this query.

EXAMPLES:

Query: "What is the roof area at 2455 New Holland Cir, Murfreesboro, TN 37128"
Analysis:
{{
    "flow": "1",
    "address": "2455 New Holland Cir, Murfreesboro, TN 37128",
    "query": "What is the roof area",
    "complete_query": "What is the roof area at 2455 New Holland Cir, Murfreesboro, TN 37128",
    "relevant_sections": ["Roof Measurements", "Diagrams"],
    "important_imagery": ["Area", "Top_View"]
}}

Query: "Find all properties with roof area greater than 2000 square feet"
Analysis:
{{
    "flow": "2",
    "address": null,
    "query": "Find all properties with roof area greater than 2000 square feet",
    "complete_query": "Find all properties with roof area greater than 2000 square feet",
    "relevant_sections": ["Roof Measurements"],
    "important_imagery": []
}}

Query: "Show me the pitch information for the property at 123 Main St, Anytown, USA"
Analysis:
{{
    "flow": "1",
    "address": "123 Main St, Anytown, USA",
    "query": "Show me the pitch information",
    "complete_query": "Show me the pitch information for the property at 123 Main St, Anytown, USA",
    "relevant_sections": ["Pitch Breakdown", "Roof Measurements", "Diagrams"],
    "important_imagery": ["Pitch_Degrees", "Pitch_on_12", "Top_View"]
}}

Query: "Which properties have more than 3 roof facets"
Analysis:
{{
    "flow": "2",
    "address": null,
    "query": "Which properties have more than 3 roof facets",
    "complete_query": "Which properties have more than 3 roof facets",
    "relevant_sections": ["House Measurements", "Roof Measurements"],
    "important_imagery": []
}}

Query: "Show me pictures of the roof at 456 Oak St"
Analysis:
{{
    "flow": "1",
    "address": "456 Oak St",
    "query": "Show me pictures of the roof",
    "complete_query": "Show me pictures of the roof at 456 Oak St",
    "relevant_sections": ["Imagery", "Diagrams"],
    "important_imagery": ["Top_View", "North_Side", "South_Side", "East_Side", "West_Side"]
}}

Query: "What is the area per pitch for this roof"
Analysis:
{{
    "flow": "1",
    "address": null,
    "query": "What is the area per pitch",
    "complete_query": "What is the area per pitch for this roof",
    "relevant_sections": ["Pitch Breakdown", "Roof Measurements", "Diagrams"],
    "important_imagery": ["Pitch_Degrees", "Pitch_on_12", "Area"]
}}

Query: "What is the waste factor for this property"
Analysis:
{{
    "flow": "1",
    "address": null,
    "query": "What is the waste factor",
    "complete_query": "What is the waste factor for this property",
    "relevant_sections": ["Waste Calculation"],
    "important_imagery": []
}}

Query: "Can you please create a report for address: 129 HIDDEN VALLEY DR, PITTSBURGH, PA 15237-1701"
Analysis:
{{
    "flow": "3",
    "address": "129 HIDDEN VALLEY DR, PITTSBURGH, PA 15237-1701",
    "query": "Can you please create a report",
    "complete_query": "Can you please create a report for address: 129 HIDDEN VALLEY DR, PITTSBURGH, PA 15237-1701",
    "relevant_sections": [],
    "important_imagery": []
}}

Now analyze this query:
{query}

Return ONLY a valid JSON object with the six fields: flow, address, query, complete_query, relevant_sections, important_imagery."""

        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "temperature": 0.1,
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

            # Parse the JSON response
            try:
                analysis_dict = json.loads(llm_response)
                analysis = QueryAnalysis(**analysis_dict)
                logger.info(f"✅ Query analyzed: Flow {analysis.flow}, Address: {analysis.address}")
                return analysis

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {llm_response}")

        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")

    
def analyze_query(query: str, region_name: str = "us-east-1") -> QueryAnalysis:
    """
    Convenience function to analyze a query.

    Args:
        query: The query to analyze
        region_name: AWS region

    Returns:
        QueryAnalysis object with flow, address, query, complete_query, and relevant_sections
    """
    router = QueryRouter(region_name=region_name)
    return router.analyze_query(query)


if __name__ == "__main__":
    # Test the query router
    test_queries = [
        "What is the roof area at 2455 New Holland Cir, Murfreesboro, TN 37128",
        "Find all properties with roof area greater than 2000 square feet",
        "Show me the pitch information for the property at 123 Main St, Anytown, USA",
        "Which properties have more than 3 roof facets",
        "When I searched for address no exact match found, even suggested address is not found in the similar address list"
    ]

    router = QueryRouter()

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        analysis = router.analyze_query(query)
        print(f"Flow: {analysis.flow}")
        print(f"Address: {analysis.address}")
        print(f"Clean Query: {analysis.query}")
        print(f"Complete Query: {analysis.complete_query}")
        print('='*80)
