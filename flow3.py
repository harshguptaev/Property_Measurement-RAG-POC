#!/usr/bin/env python3

import os
import json

from s3_utils import S3Client
from address_to_lat_long import AddressToLatLong
from pictometry_client import saveimagesfrompictometry, save_cropped_images
from llm import get_required_dimensions_from_llm
from predict_pitch import save_pitch_data
from facet_predict import run_facet_predict
from letr import save_letr_data
from draw_outline import DrawOutline
from roof_generator import RoofGenerator

def run_flow_for_address(address: str) -> dict:
    """Run the full flow for a given address and return a summary dict."""
    converter = AddressToLatLong()
    result = converter.geocode_address(address)
    lat = result["lat"]
    lon = result["lon"]
    print(f"Latitude: {lat}, Longitude: {lon}")

    # Fetch and save pictometry images
    saveimagesfrompictometry(lat, lon)

    # Use LLM to determine required crop dimensions, then crop
    dimensions = get_required_dimensions_from_llm(lat, lon)
    print(f"Dimensions: {dimensions}")
    save_cropped_images(dimensions, lat, lon)

    # # Upload cropped images to S3
    import boto3
    s3_session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_S3"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_S3"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN_S3"),
        region_name=os.getenv("AWS_REGION_S3", "us-east-2"),
    )
    s3_client = S3Client(region="us-east-2", session=s3_session)
    # s3_client.upload_cropped_images_to_s3(lat, lon)

    # # Run LETR model on top cropped image
    # save_letr_data(lat, lon)

    # Create detailed roof outline diagrams for all orientations
    lat_lon_folder = f"{lat}_{lon}"
    final_dir = os.path.join("final_data", lat_lon_folder)

    print("🎨 Creating detailed roof outline diagrams...")

    orientations = ["East", "West", "North", "South", "Top"]
    diagrams_created = 0

    for orientation in orientations:
        orientation_cap = orientation.capitalize()

        # Paths to the JSON and image files
        json_file = os.path.join(final_dir, f"{orientation_cap}_cropped_outline.json")
        image_file = os.path.join(final_dir, f"{orientation_cap}_cropped.png")
        output_dir = os.path.join(final_dir, orientation_cap.lower())

        # Check if both files exist
        if os.path.exists(json_file) and os.path.exists(image_file):
            print(f"  📐 Processing {orientation}...")
            os.makedirs(output_dir, exist_ok=True)

            # Create roof generator instance
            generator = RoofGenerator(json_file, image_file, output_dir)
            generator.load_data()

            # Generate all diagram types
            generator.create_overlay_with_yellow_borders(0.8, include_labels=True)
            generator.create_overlay_with_yellow_borders(0.8, include_labels=False)
            generator.create_simplified_outline(0.9)
            generator.create_simplified_with_lengths(0.9)
            generator.create_combined_outline_with_types(0.9)

            print(f"  ✅ {orientation} diagrams saved to {output_dir}")
            diagrams_created += 5  # 5 different diagram types per orientation
        else:
            print(f"  ⚠️ Skipping {orientation} - missing files")

    print(f"✅ Created {diagrams_created} total roof outline diagrams in final_data/{lat_lon_folder}")

    # Run pitch model; returns predominant pitch
    predominant_pitch = save_pitch_data(lat, lon)
    print(f"Predominant pitch: {predominant_pitch}")

    # Build S3 URL for Top_cropped and run facet inference pipeline
    s3_url = f"s3://evtech-us-east-2-pg-test-sunsitecomplete/property-data/LatLongData/{lat}_{lon}/Top_cropped.png"
    num_facets = run_facet_predict(lat, lon, s3_url)
    print(f"Number of facets: {num_facets}")

    # create final json
    lat_lon_folder = f"{lat}_{lon}"
    final_data_dir = os.path.join("final_data", lat_lon_folder)
    top_outline_path = os.path.join(final_data_dir, "Top_cropped_outline.json")

    # Initialize final data
    final_data = {
        "address": address,
        "latitude": lat,
        "longitude": lon,
        "predominant_pitch": predominant_pitch,
        "num_of_facets": num_facets,
        "area": None,  # Will be filled from area.json if available
        "num_of_ridges": 0,
        "num_of_eaves": 0,
        "num_of_rakes": 0,
        "num_of_valleys": 0,
        "num_of_hips": 0,
        "num_of_flashing": 0
    }
    # Try to get area from area.json
    area_json_path = os.path.join(final_data_dir, "area.json")
    if os.path.exists(area_json_path):
        try:
            with open(area_json_path, 'r') as f:
                area_data = json.load(f)
            final_data["area_json"] = area_data
            if isinstance(area_data, dict) and "total_sqft" in area_data:
                final_data["area"] = area_data["total_sqft"]
            elif isinstance(area_data, (int, float)):
                final_data["area"] = area_data
        except Exception as e:
            print(f"Warning: Could not read area from {area_json_path}: {e}")

    # Count line types from top_outline.json
    if os.path.exists(top_outline_path):
        try:
            with open(top_outline_path, 'r') as f:
                outline_data = json.load(f)

            predictions = outline_data.get("predictions", [])
            for prediction in predictions:
                line_class = prediction.get("class", "").lower()
                if line_class == "ridge":
                    final_data["num_of_ridges"] += 1
                elif line_class == "eave":
                    final_data["num_of_eaves"] += 1
                elif line_class == "rake":
                    final_data["num_of_rakes"] += 1
                elif line_class == "valley":
                    final_data["num_of_valleys"] += 1
                elif line_class == "hip":
                    final_data["num_of_hips"] += 1
                elif line_class == "flashing":
                    final_data["num_of_flashing"] += 1

            print(f"Counted line types from {top_outline_path}")
        except Exception as e:
            print(f"Warning: Could not read line counts from {top_outline_path}: {e}")
    else:
        print(f"Warning: {top_outline_path} not found, line counts will be 0")

    # Save final_data.json
    final_json_path = os.path.join(final_data_dir, "final_data.json")
    os.makedirs(final_data_dir, exist_ok=True)
    with open(final_json_path, 'w') as f:
        json.dump(final_data, f, indent=2)

    print(f"✅ Final data saved to: {final_json_path}")
    print(f"📊 Summary: {final_data['num_of_facets']} facets, {final_data['num_of_ridges']} ridges, {final_data['num_of_eaves']} eaves")

    # Create outline and outline_meta folders
    outline_dir = os.path.join(final_data_dir, "outline")
    outline_meta_dir = os.path.join(final_data_dir, "outline_meta")
    os.makedirs(outline_dir, exist_ok=True)
    os.makedirs(outline_meta_dir, exist_ok=True)

    # Copy roof_overlay_without_length.png files from each orientation folder
    orientations = ["east", "west", "north", "south", "top"]
    for orientation in orientations:
        orientation_dir = os.path.join(final_data_dir, orientation)
        source_file = os.path.join(orientation_dir, "roof_overlay_without_lengths.png")
        if os.path.exists(source_file):
            import shutil
            dest_filename = f"{orientation.capitalize()}_without_length.png"
            dest_path = os.path.join(outline_dir, dest_filename)
            shutil.copy2(source_file, dest_path)
            print(f"Copied {source_file} to {dest_path}")
        else:
            print(f"Warning: {source_file} not found")

    # Copy outline.json files to outline_meta folder
    for orientation in orientations:
        source_file = os.path.join(final_data_dir, f"{orientation.capitalize()}_cropped_outline.json")
        if os.path.exists(source_file):
            import shutil
            dest_filename = f"{orientation.capitalize()}_outline_output.json"
            dest_path = os.path.join(outline_meta_dir, dest_filename)
            shutil.copy2(source_file, dest_path)
            print(f"Copied {source_file} to {dest_path}")
        else:
            print(f"Warning: {source_file} not found")

    print(f"✅ Created outline folders and copied files")
    
    #  here upload all files to s3
    # s3_client.upload_all_files_to_s3(final_data_dir)
    
    # Run similarity search on the generated roof outline
    try:
        print("\n🔍 Running similarity search on generated roof outline...")

        # Construct S3 URL for the roof outline image
        s3_url = f"s3://evtech-us-east-2-pg-test-sunsitecomplete/property-data/LatLongData/{lat}_{lon}/top/roof_outline_simplified.png"

        # Build filter parameters from generated data
        filter_params = {
            "facet_count": num_facets,
            "predominant_pitch": predominant_pitch
        }

        # Add area if available
        if final_data.get("area"):
            filter_params["area"] = final_data["area"]

        # Import and run similarity search directly
        from run_perform_similarity_search import SimilaritySearchRunner

        # Create runner instance and run search
        runner = SimilaritySearchRunner()
        results = runner.run_search(
            s3_url=s3_url,
            filter_params=filter_params,
            perform_level2=True,
            lat_lon=f"{lat}_{lon}"
        )

        if results.get('success', False):
            print("✅ Similarity search completed successfully")
            print(f"📊 Found {len(results.get('final_results', []))} similar properties")
        else:
            print("⚠️ Similarity search completed but may have issues")
            if results.get('errors'):
                print(f"Errors: {results['errors']}")

    except Exception as e:
        print(f"⚠️ Error running similarity search: {str(e)}")

    # Return a compact summary for callers
    return {
        "address": address,
        "latitude": lat,
        "longitude": lon,
        "predominant_pitch": predominant_pitch,
        "num_of_facets": num_facets,
        "final_json_path": final_json_path,
        "final_data_dir": final_data_dir,
    }

    
    
def main():
    # address = "1291 Broad St W, Lehigh Acres, FL 33936"
    
    # 65087766
    # address = "129 HIDDEN VALLEY DR, PITTSBURGH, PA 15237-1701"
 
    # 66880741
    # address = "1304 Woodward Ct Lehigh Acres, FL 33936"

    # address = "2108 GLENWOOD DR, ARNOLD MO 63010" comparison done and in teams

    # address = "107 Laura Dr, Arnold MO 63010" comparison done and in teams

    # address = "129 HIDDEN VALLEY DR, PITTSBURGH, PA 15237-1701" comparison done and in teams

    # address = "2223 OLD LEMAY FERRY RD, ARNOLD MO 63010-2503" comparison done and in teams

    # address = "3042 Brentmoor Dr, Arnold, MO 63010-3771" not good

    # address = "125 Pinevalley Dr, Pittsburgh, PA 15229"

    # address = "2152 BLOSSOM LN, ARNOLD, MO 63010-2559"
    
    # address = " 3185 PINEBROOK DR, ARNOLD, MO 63010-3741"

    address = "4 EARLIGLOW CT, ARNOLD, MO 63010"

    # address = "2120 LONG GLEN LN, ARNOLD, MO 63010-6220"

    # address = "2560 Old Lemay Ferry Road, Arnold, MO 63010" comparison done and in teams

    # address = "3161 PINEBROOK DR, ARNOLD, MO 63010-3741" comparison done and in teams

    # address = "160 LAMP POST LN, ARNOLD, MO 63010, USA"
    
    # address = "2223 OLD LEMAY FERRY RD, ARNOLD, MO 63010"
    # Delegate to the reusable function to avoid duplication
    run_flow_for_address(address)

if __name__ == "__main__":
    main()


