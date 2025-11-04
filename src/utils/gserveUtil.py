def get_relevant_metadata_from_gserve(entire_metadata: dict) -> dict:

    required_fields =  ["geometry", 
                        "gsd", 
                        "roll", 
                        "width", 
                        "height", 
                        "bearing",
                        "altitude",
                        "elevation",
                        "camera_lat",
                        "camera_lon",
                        "declination",
                        "focal_length",
                        "focal_plane_x",
                        "focal_plane_y",
                        "bit_depth_per_channel"]

    # Filter only the keys that exist in entire_metadata
    relevant_metadata = {
        key: entire_metadata[key]
        for key in required_fields
        if key in entire_metadata
    }

    return relevant_metadata