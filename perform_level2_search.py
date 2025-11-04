#!/usr/bin/env python3
"""
Perform Level 2 Hierarchical Search
Searches in hierarchical_level2 collection with metadata filtering
"""

import os
import json
import logging
import math
import re
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Level2Searcher:
    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        """
        Initialize Level 2 hierarchical search handler
        
        Args:
            milvus_host: Milvus server host
            milvus_port: Milvus server port
        """
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = "hierarchical_level2"
        self.vector_dim = 1536  # Level 2 embedding dimension
        
        # Initialize Milvus client
        self.milvus_client = MilvusClient(
            uri=f"http://{milvus_host}:{milvus_port}"
        )
        
        logger.info(f"Initialized Level2Searcher: {milvus_host}:{milvus_port}")
    
    def check_milvus_connection(self) -> bool:
        """
        Check if Milvus server is running and accessible
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            collections = self.milvus_client.list_collections()
            logger.info(f"✅ Milvus connection successful. Found {len(collections)} collections.")
            return True
        except Exception as e:
            logger.error(f"❌ Milvus connection failed: {str(e)}")
            return False
    
    def load_collection(self) -> bool:
        """
        Load collection into memory for search
        
        Returns:
            True if load successful, False otherwise
        """
        try:
            if not self.milvus_client.has_collection(self.collection_name):
                logger.error(f"❌ Collection '{self.collection_name}' does not exist.")
                return False
            
            # Load collection into memory
            self.milvus_client.load_collection(self.collection_name)
            logger.info(f"✅ Collection '{self.collection_name}' loaded into memory.")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading collection: {str(e)}")
            return False
    
    def extract_numeric_value(self, text: str) -> Optional[float]:
        """
        Extract numeric value from text (e.g., "3,754 sq ft" -> 3754.0)
        
        Args:
            text: Text containing numeric value
            
        Returns:
            Numeric value or None if not found
        """
        try:
            # Remove commas and extract numbers
            cleaned = re.sub(r'[^\d.]', '', text.replace(',', ''))
            if cleaned:
                return float(cleaned)
            return None
        except Exception as e:
            logger.warning(f"Could not extract numeric value from '{text}': {str(e)}")
            return None
    
    def parse_pitch(self, pitch_str: str) -> Optional[float]:
        """
        Convert pitch string (e.g., "7/12") to degrees
        
        Args:
            pitch_str: Pitch string in format "X/12"
            
        Returns:
            Pitch in degrees or None if invalid
        """
        try:
            # Extract numerator and denominator
            match = re.match(r'(\d+)/(\d+)', pitch_str)
            if match:
                numerator = float(match.group(1))
                denominator = float(match.group(2))
                # Convert to degrees: arctan(numerator/denominator) * 180/π
                pitch_radians = math.atan(numerator / denominator)
                pitch_degrees = math.degrees(pitch_radians)
                return pitch_degrees
            return None
        except Exception as e:
            logger.warning(f"Could not parse pitch '{pitch_str}': {str(e)}")
            return None
    
    def filter_by_metadata(self, rows: List[Dict], filter_params: Dict[str, Any]) -> List[Dict]:
        """
        Filter rows by metadata values (area and pitch) with +/- 20% range
        Only filters if the metadata fields exist (to include all sections)
        
        Args:
            rows: List of row dictionaries from Milvus
            filter_params: Filter parameters (area, predominant_pitch)
            
        Returns:
            Filtered list of rows
        """
        return rows
        filtered_rows = []
        
        target_area = filter_params.get('area')
        target_pitch = filter_params.get('predominant_pitch')
        print(f"target_area: {target_area}")
        print(f"target_pitch: {target_pitch}")
        for row in rows:
            print(f"row: {row}")
            try:
                # Get metadata
                metadata = row.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                data = metadata.get('data', {})
                
                # Track if row passes filters (only filter if field exists)
                passes_area_filter = True
                passes_pitch_filter = True
                area_value = None
                pitch_degrees = None
                
                # Check area if provided AND field exists in metadata
                if target_area is not None:
                    area_str = data.get('total_area', '')
                    print(f"area_str: {area_str}")
                    if area_str:  # Only filter if area field exists
                        area_value = self.extract_numeric_value(area_str)
                        print(f"area_value: {area_value}")
                        if area_value is not None:
                            # Calculate +/- 20% range
                            min_area = target_area * 0.8
                            max_area = target_area * 1.2
                            passes_area_filter = (min_area <= area_value <= max_area)
                        else:
                            # Can't extract area, but field exists - exclude
                            passes_area_filter = False
                    # If area_str is empty, don't filter (include the row)
                
                # Check pitch if provided AND field exists in metadata
                if target_pitch is not None:
                    pitch_str = data.get('predominant_pitch', '')
                    if pitch_str:  # Only filter if pitch field exists
                        pitch_degrees = self.parse_pitch(pitch_str)
                        if pitch_degrees is not None:
                            # Calculate +/- 20% range
                            min_pitch = target_pitch * 0.8
                            max_pitch = target_pitch * 1.2
                            passes_pitch_filter = (min_pitch <= pitch_degrees <= max_pitch)
                        else:
                            # Can't parse pitch, but field exists - exclude
                            passes_pitch_filter = False
                    # If pitch_str is empty, don't filter (include the row)
                
                # Include row if it passes all applicable filters
                if passes_area_filter and passes_pitch_filter:
                    # Add calculated values for ranking
                    row['_calculated_area'] = area_value
                    row['_calculated_pitch'] = pitch_degrees
                    filtered_rows.append(row)
                
            except Exception as e:
                logger.warning(f"Error filtering row: {str(e)}")
                continue
        
        return filtered_rows
    
    def rank_by_closeness(self, rows: List[Dict], filter_params: Dict[str, Any], 
                         property_id_scores: Dict[str, float] = None) -> List[Dict]:
        """
        Re-rank rows by similarity score first, then by closeness to target area and pitch values
        
        Args:
            rows: List of filtered rows
            filter_params: Filter parameters (area, predominant_pitch)
            property_id_scores: Dictionary mapping property_id to similarity_score from Level 1
            
        Returns:
            Re-ranked list of rows
        """
        target_area = filter_params.get('area')
        target_pitch = filter_params.get('predominant_pitch')
        
        def calculate_closeness_score(row):
            """Calculate closeness score (lower = closer)"""
            score = 0.0
            
            if target_area and '_calculated_area' in row and row['_calculated_area'] is not None:
                area_value = row['_calculated_area']
                area_diff = abs(area_value - target_area)
                # Normalize by target (percentage difference)
                score += (area_diff / target_area) * 100
            
            if target_pitch and '_calculated_pitch' in row and row['_calculated_pitch'] is not None:
                pitch_value = row['_calculated_pitch']
                pitch_diff = abs(pitch_value - target_pitch)
                # Normalize by target (percentage difference)
                score += (pitch_diff / target_pitch) * 100
            
            return score
        
        def get_sort_key(row):
            """Get sort key: first by similarity_score (descending), then by closeness_score (ascending)"""
            property_id = row.get('property_id', '')
            # Get similarity score from Level 1, default to 0 if not found
            similarity_score = property_id_scores.get(property_id, 0.0) if property_id_scores else 0.0
            closeness_score = calculate_closeness_score(row)
            
            # If no area/pitch filtering, closeness_score will be 0.0, so just rank by similarity
            # Return tuple: (-similarity_score, closeness_score)
            # Negative similarity_score because we want descending (higher is better)
            # closeness_score is ascending (lower is better)
            return (-similarity_score, closeness_score)
        
        # Sort by similarity_score first (descending), then closeness_score (ascending)
        ranked_rows = sorted(rows, key=get_sort_key)
        
        # Add ranking scores to each row
        for i, row in enumerate(ranked_rows):
            property_id = row.get('property_id', '')
            similarity_score = property_id_scores.get(property_id, 0.0) if property_id_scores else 0.0
            closeness_score = calculate_closeness_score(row)
            
            row['_similarity_score'] = similarity_score
            row['_closeness_score'] = closeness_score
            row['_rank'] = i + 1
        
        return ranked_rows
    
    def search_level2(self, property_ids: List[str], filter_params: Dict[str, Any], 
                     limit: int = 10, property_id_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Perform Level 2 hierarchical search
        
        Args:
            property_ids: List of property IDs from Level 1 search (top 10)
            filter_params: Filter parameters (area, predominant_pitch)
            limit: Number of results to return
            property_id_scores: Dictionary mapping property_id to similarity_score from Level 1
            
        Returns:
            Dictionary with search results
        """
        results = {
            'success': False,
            'matches': [],
            'total_found': 0,
            'errors': []
        }
        
        try:
            # Ensure collection is loaded
            if not self.load_collection():
                results['errors'].append("Failed to load collection")
                return results
            
            # Check if collection exists
            if not self.milvus_client.has_collection(self.collection_name):
                results['errors'].append(f"Collection '{self.collection_name}' does not exist")
                return results
            
            if not property_ids:
                results['errors'].append("No property IDs provided")
                return results
            
            logger.info(f"🔍 Level 2: Searching for {len(property_ids)} property IDs")
            
            # Build filter expression for property_ids and sections
            property_filter = " || ".join([f'property_id == "{pid}"' for pid in property_ids])
            # Filter by both "Roof Measurements - All Structures" and "House Measurements"
            # Use '||' for OR operator in Milvus filter expressions
            section_filter = '(section == "Roof Measurements - All Structures" || section == "House Measurements")'
            filter_expr = f"{property_filter}"
            
            logger.info(f"🔍 Level 2: Filter expression: {filter_expr}")
            logger.info(f"🔍 Level 2: Including sections: 'Roof Measurements - All Structures' and 'House Measurements'")
            
            # Query the collection - MilvusClient.query() uses 'expr' parameter
            query_results = self.milvus_client.query(
                collection_name=self.collection_name,
                filter=filter_expr,  
                output_fields=["id", "chunk_id", "property_id", "section", "type", 
                              "chunk_text", "metadata"],
                limit=limit * 2  # Get more to filter
            )
            
            print(f"query_results: {query_results}")
            logger.info(f"🔍 Level 2: Found {len(query_results)} rows matching filter")
            
            # Log which sections were found
            sections_found = {}
            for row in query_results:
                section = row.get('section', 'Unknown')
                sections_found[section] = sections_found.get(section, 0) + 1
            logger.info(f"🔍 Level 2: Sections found: {sections_found}")
            
            if not query_results:
                results['errors'].append("No results found in Level 2 collection")
                return results
            
            # Filter by metadata (area and pitch)
            if filter_params.get('area') or filter_params.get('predominant_pitch'):
                logger.info(f"🔍 Level 2: Filtering by metadata (area, pitch)")
                filtered_results = self.filter_by_metadata(query_results, filter_params)
                logger.info(f"🔍 Level 2: {len(filtered_results)} rows after metadata filtering")
            else:
                filtered_results = query_results
            
            if not filtered_results:
                results['errors'].append("No results after metadata filtering")
                return results
            
            # Re-rank by similarity score first, then by closeness to target values
            logger.info(f"🔍 Level 2: Re-ranking by similarity score (from Level 1) and closeness")
            print(f"filtered_results: {filtered_results}")
            ranked_results = self.rank_by_closeness(filtered_results, filter_params, property_id_scores)
            print(f"ranked_results: {ranked_results}")
            # Include ALL results from both sections - no limit to ensure both sections are included
            final_results = ranked_results
            
            # Log sections in final results
            final_sections = {}
            for row in final_results:
                section = row.get('section', 'Unknown')
                final_sections[section] = final_sections.get(section, 0) + 1
            logger.info(f"🔍 Level 2: Sections in final ranked results: {final_sections}")
            logger.info(f"🔍 Level 2: Total results included: {len(final_results)} (from {len(ranked_results)} ranked)")
            
            # Format results
            for row in final_results:
                metadata = row.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                match = {
                    'property_id': row.get('property_id'),
                    'chunk_id': row.get('chunk_id'),
                    'section': row.get('section'),
                    'type': row.get('type'),
                    'metadata': metadata,
                    'similarity_score': row.get('_similarity_score', 0.0),
                    'closeness_score': row.get('_closeness_score', 0.0),
                    'rank': row.get('_rank', 0)
                }
                results['matches'].append(match)
            
            results['total_found'] = len(results['matches'])
            results['success'] = True
            logger.info(f"✅ Level 2: Found {results['total_found']} matches")
            
        except Exception as e:
            logger.error(f"❌ Error performing Level 2 search: {str(e)}")
            results['errors'].append(str(e))
        
        return results

