# This file will take an array of report ids, and for each report it will call the report_client file to download the pdf and save it to input_data/Report-<report_id>/<report_id>.pdf
# Than it will call for each of the report pdf in input_data/Report-<report_id>/<report_id>.pdf the llm for calculatingthe chunk in this form whichever data is present give that only and save the data to output_data/Report-<report_id>/<report_id>.json
#   {
#     "property_id": "PROP_67668772",
#     "address": "3374 Victoria Ave, Santa Clara, CA 95051",
#     "latitude": 37.3565098,
#     "longitude": -121.9879742,
#     "owner": ""
#   },
#   {
#     "chunk_id": "C001",
#     "property_id": "PROP_67668772",
#     "section": "House Measurements",
#     "type": "text",
#     "data": {
#       "number_of_stories": "<=1",
#       "total_roof_facets": 11,
#       "structure_complexity": "Normal",
#       "estimated_attic": "2418.8 sq ft",
#       "total_roof_obstructions": 9
#     }
#   },
#   {
#     "chunk_id": "C002",
#     "property_id": "PROP_67668772",
#     "section": "Roof Measurements - All Structures",
#     "type": "text",
#     "data": {
#       "structure": "All Structures",
#       "total_area": "2607.2 sq ft",
#       "total_roof_facets": 11,
#       "predominant_pitch": "5/12",
#       "ridges": "58 ft (3 Ridges)",
#       "hips": "136.3 ft (8 Hips)",
#       "valleys": "86.8 ft (6 Valleys)",
#       "rakes": "15.6 ft (3 Rakes)",
#       "eaves_starter": "211.1 ft (8 Eaves)",
#       "drip_edge": "226.7 ft (11 Lengths)",
#       "flashing": "7.1 ft (2 Lengths)",
#       "step_flashing": "5.9 ft (3 Lengths)",
#       "parapet_walls": "0 ft (0 Lengths)",
#       "roof_obstructions_perimeter": "37.2 ft",
#       "roof_obstructions_area": "11 sq ft",
#       "net_roof_area": "2596.2 sq ft"
#     }
#   },

# code start


from reportExtractor import export_data_for_report
from chunking import export_chunk_data_for_report
from image_processing import preprocess_image
from run_create_embeddings import create_embeddings_for_report, create_hierarchical_embeddings_for_report
from s3_utils import upload_ddd_to_s3

def start_chunking():
    # report_ids = [64053749,64048796,64046823,64045243,64043072,64024785, 64023965, 64020821, 64019882, 64017010, 64016168, 64010425, 64008455, 64007311]
    # report_ids = [64024785, 64023965, 64020821, 64019882, 64017010, 64016168, 64012843, 64010843, 64010574, 64010425, 64008455, 64007311]
    # report_ids = [66758308, 64895822, 64892357, 64010574, 64010843, 64012843]
    report_ids = [58095818]
    print(f"Starting to process {len(report_ids)} reports: {report_ids}")

    for i, report_id in enumerate(report_ids, 1):
        print(f"Processing report {i}/{len(report_ids)}: {report_id}")
        try:
            export_data_for_report(str(report_id))
            export_chunk_data_for_report(str(report_id))
            print(f"✅ Completed report {report_id}")

            # Preprocess image (dd.png or similar)
            # Look for image to preprocess in the report directory
            print(f"Preprocessing image for report {report_id}")
            preprocess_image(str(report_id))
            print(f"✅ Image preprocessing completed for report {report_id}")

            # upload DDD to s3 bucket (not required may be)
            s3_url = upload_ddd_to_s3(str(report_id))
            print(f"DDD uploaded to s3 bucket: {s3_url}")

            # Create embeddings for the processed image
            create_embeddings_for_report(str(report_id))

            # Create hierarchical embeddings for the chunks (Level 2)
            create_hierarchical_embeddings_for_report(str(report_id))

        except Exception as e:
            print(f"❌ Error processing report {report_id}: {e}")
            
    
    print("All reports processed!")


start_chunking();