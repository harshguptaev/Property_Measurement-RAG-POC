"""
Query Router for Property Measurement RAG System

This module analyzes user queries and determines the appropriate search flow:
- Flow 1: Property-specific queries (address given, find specific information)
- Flow 2: Non-property-specific queries (find properties matching criteria)
"""

import json
import logging
from typing import Dict, Any, Optional
import boto3
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class QueryAnalysis(BaseModel):
    """Structured analysis of a user query."""
    flow: str  # "1" or "2"
    address: Optional[str] = None  # Extracted address if present
    query: str  # Cleaned user intention/query
    complete_query: str  # Full original query


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
            QueryAnalysis object with flow, address, query, and complete_query
        """
        prompt = f"""You are a query analysis expert for a property measurement and roofing analysis system.

Your task is to analyze the user's query and extract the following information:

1. **flow**: Determine if this is:
   - "1": Property-specific query (user mentions a specific address/property and wants information about it)
   - "2": Non-property-specific query (user wants to find properties that match certain criteria, like "find properties with area > 2000 sq ft")

2. **address**: If the query mentions a specific address, extract it exactly as written. If no address is mentioned, set to null.

3. **query**: The user's actual intent/question, cleaned of address information if present.

4. **complete_query**: The full original query as provided.

EXAMPLES:

Query: "What is the roof area at 2455 New Holland Cir, Murfreesboro, TN 37128"
Analysis:
{{
    "flow": "1",
    "address": "2455 New Holland Cir, Murfreesboro, TN 37128",
    "query": "What is the roof area",
    "complete_query": "What is the roof area at 2455 New Holland Cir, Murfreesboro, TN 37128"
}}

Query: "Find all properties with roof area greater than 2000 square feet"
Analysis:
{{
    "flow": "2",
    "address": null,
    "query": "Find all properties with roof area greater than 2000 square feet",
    "complete_query": "Find all properties with roof area greater than 2000 square feet"
}}

Query: "Show me the pitch information for the property at 123 Main St, Anytown, USA"
Analysis:
{{
    "flow": "1",
    "address": "123 Main St, Anytown, USA",
    "query": "Show me the pitch information",
    "complete_query": "Show me the pitch information for the property at 123 Main St, Anytown, USA"
}}

Query: "Which properties have more than 3 roof facets"
Analysis:
{{
    "flow": "2",
    "address": null,
    "query": "Which properties have more than 3 roof facets",
    "complete_query": "Which properties have more than 3 roof facets"
}}

Now analyze this query:
{query}

Return ONLY a valid JSON object with the four fields: flow, address, query, complete_query."""

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
        QueryAnalysis object
    """
    router = QueryRouter(region_name=region_name)
    return router.analyze_query(query)


if __name__ == "__main__":
    # Test the query router
    test_queries = [
        "What is the roof area at 2455 New Holland Cir, Murfreesboro, TN 37128",
        "Find all properties with roof area greater than 2000 square feet",
        "Show me the pitch information for the property at 123 Main St, Anytown, USA",
        "Which properties have more than 3 roof facets"
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
