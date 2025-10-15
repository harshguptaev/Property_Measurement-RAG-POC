import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


def read_pdf_text_by_page(pdf_path: str) -> List[str]:
    import fitz
    doc = fitz.open(pdf_path)
    pages = []
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text")
            pages.append(text or "")
    finally:
        doc.close()
    return pages


def _extract_premium_header(text: str) -> Dict[str, Any]:
    """Extract header information from premium report."""
    data = {}

    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', text)
    if date_match:
        data['date'] = date_match.group(1)

    address_match = re.search(r'(\d+[^,\n]+,[^,\n]+,\s*[A-Z]{2}\s+\d{5})', text)
    if address_match:
        data['property_address'] = address_match.group(1).strip()

    report_match = re.search(r'Report:\s*(\d+)', text)
    if report_match:
        data['report_id'] = report_match.group(1)

    return data


def _extract_prepared_for_premium(text: str) -> Optional[Dict[str, Any]]:
    """Extract prepared for information from premium report."""
    prepared_section = re.search(r'PREPARED FOR\s*(.*?)(?=TABLE OF CONTENTS|MEASUREMENTS)', text, re.DOTALL | re.IGNORECASE)
    if not prepared_section:
        return None
    
    section_text = prepared_section.group(1)
    data = {}
    
    contact_match = re.search(r'Contact:\s*([^\n]+)', section_text)
    if contact_match:
        data['contact'] = contact_match.group(1).strip()
    
    company_match = re.search(r'Company:\s*([^\n]+)', section_text)
    if company_match:
        data['company'] = company_match.group(1).strip()
    
    address_match = re.search(r'Address:\s*([^\n]+(?:\n[^\n:]+)*?)(?=\s*Phone:|$)', section_text, re.DOTALL)
    if address_match:
        address_lines = [line.strip() for line in address_match.group(1).split('\n') if line.strip()]
        data['address'] = ', '.join(address_lines)
    
    phone_match = re.search(r'Phone:\s*([0-9-]+)', section_text)
    if phone_match:
        data['phone'] = phone_match.group(1)
    
    return data


def _extract_summary_measurements(text: str) -> Dict[str, Any]:
    """Extract summary measurements from the first page."""
    data = {}
    
    patterns = {
        'total_roof_area': r'Total Roof Area\s*=\s*([0-9,]+\s*sq ft)',
        'total_roof_facets': r'Total Roof Facets\s*=\s*(\d+)',
        'predominant_pitch': r'Predominant Pitch\s*=\s*([0-9/]+)',
        'number_of_stories': r'Number of Stories\s*([<>=]*\d+)',
        'total_ridges_hips': r'Total Ridges/Hips\s*=\s*([0-9]+\s*ft)',
        'total_valleys': r'Total Valleys\s*=\s*([0-9]+\s*ft)',
        'total_rakes': r'Total Rakes\s*=\s*([0-9]+\s*ft)',
        'total_eaves': r'Total Eaves\s*=\s*([0-9]+\s*ft)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data[key] = match.group(1).strip()
    
    return data


def _extract_detailed_measurements(text: str) -> Dict[str, Any]:
    """Extract detailed measurements from report summary page."""
    data = {}
    
    patterns = {
        'ridges': r'Ridges\s*=\s*([0-9]+\s*ft\s*\([^)]+\))',
        'hips': r'Hips\s*=\s*([0-9]+\s*ft\s*\([^)]+\))',
        'valleys': r'Valleys\s*=\s*([0-9]+\s*ft\s*\([^)]+\))',
        'rakes': r'Rakes†?\s*=\s*([0-9]+\s*ft\s*\([^)]+\))',
        'eaves_starter': r'Eaves/Starter‡?\s*=\s*([0-9]+\s*ft\s*\([^)]+\))',
        'drip_edge': r'Drip Edge[^=]*=\s*([0-9]+\s*ft\s*\([^)]+\))',
        'parapet_walls': r'Parapet Walls\s*=\s*([0-9]+\s*\([^)]+\))',
        'flashing': r'Flashing\s*=\s*([0-9]+\s*ft\s*\([^)]+\))',
        'step_flashing': r'Step flashing\s*=\s*([0-9]+\s*ft\s*\([^)]+\))',
        'predominant_pitch': r'Predominant Pitch\s*=\s*([0-9/]+)',
        'total_area_all_pitches': r'Total Area \(All Pitches\)\s*=\s*([0-9,]+\s*sq ft)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data[key] = match.group(1).strip()
    
    return data


def _extract_pitch_breakdown(text: str) -> List[Dict[str, Any]]:
    """Extract pitch breakdown table from report summary."""
    pitch_data = []
    
    # Look for the pitch breakdown section more specifically
    pitch_section = re.search(r'Roof Pitches\s*Area \(sq ft\)\s*% of Roof\s*(.*?)(?=The table above|Waste Calculation)', text, re.DOTALL | re.IGNORECASE)
    if pitch_section:
        content = pitch_section.group(1).strip()
        # Split into lines and process each line
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Look for lines with pitch format (e.g., "6/12", "8/12", etc.)
        for line in lines:
            # Match lines with pitch/area/percentage format
            match = re.match(r'(\d+/\d+)\s+([\d.]+)\s+([\d.]+%)', line)
            if match:
                pitch_data.append({
                    'pitch': match.group(1),
                    'area_sq_ft': match.group(2),
                    'percentage': match.group(3)
                })
    
    return pitch_data


def _extract_waste_calculation(text: str) -> List[Dict[str, Any]]:
    """Extract waste calculation table."""
    waste_data = []
    
    # Look for the waste calculation table more specifically
    waste_section = re.search(r'Waste %.*?Area \(sq ft\).*?Squares\s*(.*?)(?=This table|All Structures)', text, re.DOTALL | re.IGNORECASE)
    if waste_section:
        content = waste_section.group(1).strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            # Match lines with percentage/area/squares format
            match = re.match(r'(\d+%)\s+([\d,]+)\s+([\d.]+)', line)
            if match:
                waste_data.append({
                    'waste_percentage': match.group(1),
                    'area_sq_ft': match.group(2),
                    'squares': match.group(3)
                })
    
    return waste_data


def _extract_coordinates(text: str) -> Dict[str, Any]:
    """Extract property coordinates."""
    data = {}
    
    longitude_match = re.search(r'Longitude\s*=\s*([-0-9.]+)', text)
    if longitude_match:
        data['longitude'] = longitude_match.group(1)
    
    latitude_match = re.search(r'Latitude\s*=\s*([-0-9.]+)', text)
    if latitude_match:
        data['latitude'] = latitude_match.group(1)
    
    return data


def _extract_online_maps(text: str) -> Dict[str, Any]:
    """Extract online maps and directions links."""
    data = {}

    # Extract property map URL
    property_map_match = re.search(r'Online map of property\s*(http[^\s]+)', text)
    if property_map_match:
        data['property_map_url'] = property_map_match.group(1)

    # Extract directions URL
    directions_match = re.search(r'Directions from.*?\s*(http[^\s]+)', text, re.DOTALL)
    if directions_match:
        data['directions_url'] = directions_match.group(1)

    return data


def _extract_business_links(text: str) -> Dict[str, str]:
    """Extract business links from the premium report."""
    business_links = {}

    # Patterns for different business types
    patterns = {
        'restaurants': r'Restaurants\s*http[^\s]+',
        'fast_food': r'Fast Food\s*http[^\s]+',
        'medical_centers': r'Medical Centers\s*http[^\s]+',
        'hospitals': r'Hospitals\s*http[^\s]+',
        'doctors': r'Doctors\s*http[^\s]+',
        'gas_stations': r'Gas Stations\s*http[^\s]+'
    }

    for business_type, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            url_match = re.search(r'http[^\s]+', match.group(0))
            if url_match:
                business_links[business_type] = url_match.group(0)

    return business_links


def extract_premium_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract important text chunks from premium PDF format."""

    report_id = Path(pdf_path).stem
    if '_Premium' in report_id:
        report_id = report_id.replace('_Premium', '')
    # Extract the numeric report ID from filename like "report_67849245"
    report_id_match = re.search(r'report_(\d+)', report_id)
    if report_id_match:
        report_id = report_id_match.group(1)

    pages = read_pdf_text_by_page(pdf_path)
    all_text = "\n".join(pages)

    chunks: List[Dict[str, Any]] = []

    def _add(section: str, _type: str, data: Dict[str, Any]):
        chunk_id = f"{report_id}_chunk_{len(chunks)+1}"
        chunks.append({
            "chunk_id": chunk_id,
            "section": section,
            "type": _type,
            "data": data
        })

    header_data = _extract_premium_header(all_text)
    if header_data:
        _add("Report Header", "text", header_data)

    online_maps = _extract_online_maps(all_text)
    if online_maps:
        _add("Property Navigation", "text", online_maps)

    prepared_for = _extract_prepared_for_premium(all_text)
    if prepared_for:
        _add("Prepared For", "text", prepared_for)

    summary_measurements = _extract_summary_measurements(all_text)
    if summary_measurements:
        _add("Summary Measurements", "text", summary_measurements)

    detailed_measurements = _extract_detailed_measurements(all_text)
    if detailed_measurements:
        _add("Detailed Measurements", "text", detailed_measurements)

    coordinates = _extract_coordinates(all_text)
    if coordinates:
        _add("Property Location", "text", coordinates)

    business_links = _extract_business_links(all_text)
    if business_links:
        _add("Business Links", "text", {
            "description": "Links to businesses near the property including restaurants, fast food, medical centers, hospitals, doctors, and gas stations",
            "links": business_links
        })

    return chunks


def write_premium_chunks_output(pdf_path: str, chunks: List[Dict[str, Any]]) -> Path:
    """Write premium chunks to output files."""
    stem = Path(pdf_path).stem

    # Extract the numeric report ID from filename like "report_67849245"
    report_id_match = re.search(r'report_(\d+)', stem)
    if report_id_match:
        report_id = report_id_match.group(1)
    else:
        report_id = stem

    out_dir = Path("docling_exports") / f"{report_id}_Premium"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "important_chunks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    final_chunks_dir = Path("Final_Chunks")
    final_chunks_dir.mkdir(parents=True, exist_ok=True)
    final_chunks_file = final_chunks_dir / f"RoofReport-{report_id}.json"

    final_chunks_data = {
        "text": chunks
    }

    with open(final_chunks_file, "w", encoding="utf-8") as f:
        json.dump(final_chunks_data, f, ensure_ascii=False, indent=2)

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Extract chunks from Premium PDF reports")
    parser.add_argument("--pdf", required=True, help="Path to premium PDF file")
    parser.add_argument("--print", action="store_true", help="Print chunks to stdout")
    args = parser.parse_args()
    
    pdf_path = os.path.abspath(args.pdf)
    chunks = extract_premium_chunks(pdf_path)
    out_file = write_premium_chunks_output(pdf_path, chunks)
    
    if args.print:
        print(json.dumps(chunks, ensure_ascii=False, indent=2))
    
    print(f"Premium chunks extracted to: {out_file}")


if __name__ == "__main__":
    main()
