#!/usr/bin/env python3
"""
Run Similarity Search
Main script to perform similarity search on roof outline images
"""

import os
import sys
import argparse
import json
import logging
import re
import math
from typing import Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from perform_similarity_search import SimilaritySearcher
from perform_level2_search import Level2Searcher
from final_llm_result_generator import FinalLLMResultGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimilaritySearchRunner:
    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        """
        Initialize similarity search runner with both Level 1 and Level 2 searchers
        
        Args:
            milvus_host: Milvus server host
            milvus_port: Milvus server port
        """
        self.level1_searcher = SimilaritySearcher(milvus_host, milvus_port)
        self.level2_searcher = Level2Searcher(milvus_host, milvus_port)
    
    def run_search(self, s3_url: str, filter_params: dict = None,
                   perform_level2: bool = True, lat_lon: str = None) -> dict:
        """
        Run two-level hierarchical similarity search with S3 URL
        
        Args:
            s3_url: S3 URL of the roof_outline_simplified.png image
            filter_params: Optional filter parameters (facet_count, area, predominant_pitch, etc.)
            perform_level2: Whether to perform Level 2 search (default: True)
            
        Returns:
            Dictionary with combined search results
        """
        print("\n🔍 Roof Imagery RAG - Hierarchical Similarity Search")
        print("=" * 60)
        print(f"S3 URL: {s3_url}")
        if filter_params:
            print(f"Filter Parameters: {json.dumps(filter_params, indent=2)}")
        print()
        
        combined_results = {
            'success': False,
            'level1': {},
            'level2': {},
            'final_results': [],
            'errors': []
        }
        
        # Check Milvus connection
        print("🔗 Step 1: Checking Milvus Connection")
        if not self.level1_searcher.check_milvus_connection():
            print("❌ Failed to connect to Milvus. Please ensure Milvus is running.")
            combined_results['errors'].append('Milvus connection failed')
            return combined_results
        print("✅ Milvus connection successful")
        
        # Perform Level 1 search (get top 5, but we need top 10 for level 2)
        print("\n🔍 Step 2: Level 1 - Performing Similarity Search")
        level1_results = self.level1_searcher.search_by_s3_url(
            s3_url, 
            limit=5,  # Top 5 for final results
            filter_params=filter_params
        )
        combined_results['level1'] = level1_results
        
        if not level1_results['success']:
            print("❌ Level 1 search failed")
            combined_results['errors'].extend(level1_results.get('errors', []))
            return combined_results
        
        print(f"✅ Level 1: Found {level1_results['total_found']} similar roofs (after score filtering >= 0.80)")
        
        # Display Level 1 results
        if level1_results['matches']:
            print("\n📊 Level 1 Results (Top 5, Score >= 0.80):")
            print("-" * 60)
            for i, match in enumerate(level1_results['matches'], 1):
                print(f"Rank {i}: Property ID: {match['property_id']}, "
                      f"Address: {match.get('address', 'N/A')}, "
                      f"Score: {match['similarity_score']:.4f}")
        else:
            print("⚠️ No results passed the similarity score threshold (>= 0.80)")
            print("   Skipping Level 2 search as there are no property IDs to search.")
        
        # Perform Level 2 search if enabled and we have results
        if perform_level2 and level1_results['matches']:
            print("\n🔍 Step 3: Level 2 - Performing Hierarchical Search")
            
            # Get top 10 property IDs and their similarity scores for Level 2
            # Create a mapping of property_id -> similarity_score
            property_id_scores = {match['property_id']: match['similarity_score'] 
                                 for match in level1_results['matches']}
            property_ids = list(property_id_scores.keys())
            
            # If we need more, do another search with limit=10
            if len(property_ids) < 10:
                level1_extended = self.level1_searcher.search_by_s3_url(
                    s3_url,
                    limit=10,
                    filter_params=filter_params
                )
                if level1_extended['success']:
                    # Update the mapping with extended results
                    for match in level1_extended['matches'][:10]:
                        property_id_scores[match['property_id']] = match['similarity_score']
                    property_ids = list(property_id_scores.keys())[:10]
            
            print(f"🔍 Level 2: Searching with {len(property_ids)} property IDs")
            
            level2_results = self.level2_searcher.search_level2(
                property_ids=property_ids,
                property_id_scores=property_id_scores,  # Pass similarity scores
                filter_params=filter_params or {},
                limit=10
            )
            combined_results['level2'] = level2_results
            
            if level2_results['success']:
                print(f"✅ Level 2: Found {level2_results['total_found']} matches")
                
                # Display Level 2 results
                if level2_results['matches']:
                    print("\n📊 Level 2 Results (Ranked by Similarity Score, then Closeness):")
                    print("-" * 60)
                    for match in level2_results['matches']:
                        metadata_data = match.get('metadata', {}).get('data', {})
                        section = match.get('section', 'N/A')
                        print(f"Rank {match.get('rank', 'N/A')}: Property ID: {match['property_id']}, "
                              f"Section: {section}, "
                              f"Similarity Score: {match.get('similarity_score', 0):.4f}, "
                              f"Closeness Score: {match.get('closeness_score', 0):.2f}, "
                              f"Area: {metadata_data.get('total_area', 'N/A')}, "
                              f"Pitch: {metadata_data.get('predominant_pitch', 'N/A')}")
                
                # Combine results: Level 2 results are the final ranking
                combined_results['final_results'] = level2_results['matches']
                combined_results['success'] = True
            else:
                print("⚠️ Level 2 search failed, using Level 1 results")
                combined_results['final_results'] = level1_results['matches']
                combined_results['success'] = level1_results['success']
                combined_results['errors'].extend(level2_results.get('errors', []))
        else:
            # No Level 2, use Level 1 results
            combined_results['final_results'] = level1_results['matches']
            combined_results['success'] = level1_results['success']
        
        # Collect top result (rank 1) details from both level1 and level2
        if combined_results.get('final_results'):
            top_result = self._collect_top_result_details(combined_results)
            if top_result:
                combined_results['top_result'] = top_result

        # Save results to file (needed for LLM summary generation)
        output_file = "similarity_search_results.json"
        with open(output_file, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_file}")

        # Generate LLM summary if final_data.json exists
        if lat_lon:
            self._generate_llm_summary(combined_results, lat_lon)

        return combined_results
    
    def _collect_top_result_details(self, combined_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Collect all details from level1 and level2 for the rank 1 property
        
        Args:
            combined_results: The combined results dictionary
            
        Returns:
            Dictionary with top result details or None if no rank 1 found
        """
        try:
            final_results = combined_results.get('final_results', [])
            if not final_results:
                return None
            
            # Find rank 1 result
            rank_1_result = None
            for result in final_results:
                if result.get('rank') == 1:
                    rank_1_result = result
                    break
            
            if not rank_1_result:
                # If no rank found, use first result
                rank_1_result = final_results[0]
            
            top_property_id = rank_1_result.get('property_id')
            if not top_property_id:
                return None
            
            # Collect level1 details for this property
            level1_match = None
            level1_results = combined_results.get('level1', {}).get('matches', [])
            for match in level1_results:
                if match.get('property_id') == top_property_id:
                    level1_match = match
                    break
            
            # Collect all level2 entries for this property
            level2_matches = []
            level2_results = combined_results.get('level2', {}).get('matches', [])
            for match in level2_results:
                if match.get('property_id') == top_property_id:
                    level2_matches.append(match)
            
            # Build top_result object
            top_result = {
                'property_id': top_property_id,
                'rank': rank_1_result.get('rank', 1),
                'level1_details': level1_match if level1_match else None,
                'level2_details': level2_matches if level2_matches else []
            }
            
            logger.info(f"✅ Collected top result details for property: {top_property_id}")
            logger.info(f"   Level1 entries: {1 if level1_match else 0}")
            logger.info(f"   Level2 entries: {len(level2_matches)}")
            
            return top_result

        except Exception as e:
            logger.error(f"❌ Error collecting top result details: {str(e)}")
            return None

    def _generate_llm_summary(self, combined_results: Dict[str, Any], lat_lon: str):
        """
        Generate LLM summary using final_data.json and similarity results
        """
        try:
            # Use absolute paths based on where files are actually saved
            similarity_results_path = os.path.abspath("similarity_search_results.json")

            # Use final_data.json from the lat_lon folder
            house2_summary_path = os.path.abspath(f"final_data/{lat_lon}/final_data.json")
            print(f"Looking for final_data.json at: {house2_summary_path}")

            if os.path.exists(house2_summary_path):
                print("\n🤖 Generating LLM Summary...")
                print("=" * 60)

                # Import here to avoid circular imports
                from final_llm_result_generator import FinalLLMResultGenerator

                # Use absolute paths for files
                generator = FinalLLMResultGenerator(
                    similarity_results_file=similarity_results_path,
                    house2_summary_file=house2_summary_path,
                    bedrock_region="us-east-1",
                    bedrock_model="anthropic.claude-3-5-sonnet-20240620-v1:0"
                )

                # Generate comparison
                comparison_result = generator.generate_comparison()

                # Save comparison JSON
                comparison_output = os.path.abspath("final_comparison_result.json")
                generator.save_comparison(comparison_result, comparison_output)

                # Generate LLM prompt
                prompt = generator.generate_llm_prompt(comparison_result)

                # Call LLM
                llm_summary = generator.call_llm(prompt)

                # Save LLM summary
                llm_output_path = os.path.abspath("llm_summary_result.txt")
                generator.save_llm_summary(llm_summary, llm_output_path)

                print(f"\n✅ LLM Summary Generated")
                print(f"💾 Comparison JSON saved to: {comparison_output}")
                print(f"💾 LLM summary saved to: {llm_output_path}")
                print(f"\n📝 Summary Preview (first 500 characters):")
                print("-" * 60)
                print(llm_summary[:500] + "..." if len(llm_summary) > 500 else llm_summary)

            else:
                print(f"\n⚠️ {house2_summary_path} not found. Skipping LLM summary generation.")

        except Exception as e:
            logger.warning(f"⚠️ LLM summary generation failed: {str(e)}")
            print(f"⚠️ LLM summary generation failed: {str(e)}")
            print("   Continuing without LLM summary...")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Perform two-level hierarchical similarity search on roof outline images using Milvus"
    )
    parser.add_argument(
        "s3_url",
        nargs="?",
        help="S3 URL of the roof_outline_simplified.png image (e.g., s3://bucket/path/to/image.png)"
    )
    parser.add_argument(
        "--filter-params",
        type=str,
        help="JSON string with filter parameters: {\"facet_count\": 10, \"area\": 3754, \"predominant_pitch\": \"7/12\", \"ridge_count\": 5, \"eave_count\": 8}"
    )
    parser.add_argument(
        "--filter-params-file",
        type=str,
        help="Path to JSON file with filter parameters"
    )
    parser.add_argument(
        "--no-level2",
        action="store_true",
        help="Skip Level 2 hierarchical search (only perform Level 1)"
    )
    parser.add_argument(
        "--milvus-host",
        default="localhost",
        help="Milvus server host (default: localhost)"
    )
    parser.add_argument(
        "--milvus-port",
        type=int,
        default=19530,
        help="Milvus server port (default: 19530)"
    )
    parser.add_argument(
        "--lat-lon",
        type=str,
        help="Latitude and longitude in format 'lat_lon' to locate final_data.json"
    )
    
    args = parser.parse_args()
    
    # Example S3 URL if not provided
    if not args.s3_url:
        print("⚠️  No S3 URL provided. Using example URL.")
        example_url = "s3://evtech-us-east-2-pg-test-sunsitecomplete/property-data/image_query/27.758872_-82.666503/Top/roof_outline_simplified.png"
        print(f"Example: {example_url}\n")
        use_example = input("Use example URL? (y/n): ").strip().lower()
        if use_example == 'y':
            args.s3_url = example_url
        else:
            print("Please provide an S3 URL as argument:")
            print("python run_perform_similarity_search.py s3://bucket/path/to/image.png")
            return
    
    # Parse filter parameters
    filter_params = None
    if args.filter_params_file:
        try:
            with open(args.filter_params_file, 'r') as f:
                filter_params = json.load(f)
        except Exception as e:
            print(f"❌ Error reading filter params file: {str(e)}")
            return
    elif args.filter_params:
        try:
            filter_params = json.loads(args.filter_params)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing filter params JSON: {str(e)}")
            return
    
    # Example filter params structure for reference
    if filter_params:
        # Convert predominant_pitch from string to degrees if provided
        if 'predominant_pitch' in filter_params and isinstance(filter_params['predominant_pitch'], str):
            # Parse pitch string (e.g., "7/12") to degrees
            match = re.match(r'(\d+)/(\d+)', filter_params['predominant_pitch'])
            if match:
                numerator = float(match.group(1))
                denominator = float(match.group(2))
                pitch_radians = math.atan(numerator / denominator)
                filter_params['predominant_pitch'] = math.degrees(pitch_radians)
    
    # Initialize and run search
    runner = SimilaritySearchRunner(args.milvus_host, args.milvus_port)
    results = runner.run_search(
        args.s3_url, 
        filter_params=filter_params,
        perform_level2=not args.no_level2
    )
    
if __name__ == "__main__":
    main()

