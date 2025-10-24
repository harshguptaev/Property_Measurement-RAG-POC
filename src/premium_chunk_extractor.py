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
    prepared_section = re.search(r'Prepared for\s*(.*?)(?=September \d{1,2}, \d{4}|Property Address:)', text, re.DOTALL | re.IGNORECASE)
    if not prepared_section:
        return None

    section_text = prepared_section.group(1)
    lines = [line.strip() for line in section_text.split('\n') if line.strip()]

    if len(lines) >= 4:
        data = {}
        data['contact'] = f"{lines[0]} {lines[1]}" if len(lines) > 1 else lines[0]
        data['address'] = ', '.join(lines[2:-1]) if len(lines) > 3 else lines[2] if len(lines) > 2 else ""
        data['phone'] = lines[-1] if lines else ""
        return data

    return None


def _extract_summary_measurements(text: str) -> Dict[str, Any]:
    """Extract summary measurements from the measurements section."""
    data = {}

    patterns = {
        'total_roof_area': r'Area:\s*([0-9,]+\.?\d*\s*sq ft)',
        'total_roof_facets': r'Roof Facets:\s*(\d+)',
        'predominant_pitch': r'Predominant Pitch:\s*([0-9°]+)',
        'number_of_stories': r'Number of Stories:\s*([<>=]*\d+)',
        'total_ridges_hips': r'Ridges/Hips:\s*([0-9]+\.?\d*\s*ft)',
        'total_valleys': r'Valleys:\s*([0-9]+\.?\d*\s*ft)',
        'total_rakes': r'Rakes:\s*([0-9]+\.?\d*\s*ft)',
        'total_eaves': r'Eaves:\s*([0-9]+\.?\d*\s*ft)',
        'estimated_attic': r'Estimated Attic:\s*([0-9,]+\.?\d*\s*sq ft)',
        'total_roof_obstructions': r'Roof Obstructions:\s*(\d+)',
        'roof_obstructions_perimeter': r'Roof Obstructions Perimeter:\s*([0-9]+\.?\d*\s*ft)',
        'roof_obstructions_area': r'Roof Obstructions Area:\s*([0-9]+\.?\d*\s*sq ft)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data[key] = match.group(1).strip()

    return data


def _extract_detailed_measurements(text: str) -> Dict[str, Any]:
    """Extract detailed measurements from the lengths section."""
    data = {}

    patterns = {
        'ridges': r'Ridges\s*=\s*([0-9]+\.?\d*\s*ft)',
        'hips': r'Hips\s*=\s*([0-9]+\.?\d*\s*ft)',
        'valleys': r'Valleys\s*=\s*([0-9]+\.?\d*\s*ft)',
        'rakes': r'Rakes\s*=\s*([0-9]+\.?\d*\s*ft)',
        'eaves_starter': r'Eaves\s*=\s*([0-9]+\.?\d*\s*ft)',
        'flashing': r'Flashing\s*=\s*([0-9]+\.?\d*\s*ft)',
        'step_flashing': r'Step flashing\s*=\s*([0-9]+\.?\d*\s*ft)',
        'parapet_walls': r'Parapets\s*=\s*([0-9]+\.?\d*\s*ft)',
        'predominant_pitch': r'Predominant Pitch:\s*([0-9°]+)',
        'total_area_all_pitches': r'Area:\s*([0-9,]+\.?\d*\s*sq ft)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key in ['ridges', 'hips', 'valleys', 'rakes', 'eaves_starter', 'flashing', 'step_flashing', 'parapet_walls']:
                # Convert to format like "58 ft (3 Ridges)" - we'll need to count them from the detailed breakdown
                data[key] = f"{value} (Multiple {key.title()})"
            else:
                data[key] = value

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

    property_id = f"PROP_{report_id}"

    pages = read_pdf_text_by_page(pdf_path)
    all_text = "\n".join(pages)

    chunks: List[Dict[str, Any]] = []

    def _add(chunk_id: str, property_id: str, section: str, _type: str, data: Dict[str, Any]):
        chunks.append({
            "chunk_id": chunk_id,
            "property_id": property_id,
            "section": section,
            "type": _type,
            "data": data
        })

    header_data = _extract_premium_header(all_text)
    coordinates = _extract_coordinates(all_text)
    prepared_for = _extract_prepared_for_premium(all_text)

    property_info = {
        "property_id": property_id,
        "address": header_data.get("property_address", ""),
        "latitude": float(coordinates.get("latitude", 0)),
        "longitude": float(coordinates.get("longitude", 0)),
        "owner": prepared_for.get("contact", "") if prepared_for else ""
    }
    chunks.append(property_info)

    summary_measurements = _extract_summary_measurements(all_text)
    detailed_measurements = _extract_detailed_measurements(all_text)

    if summary_measurements:
        house_data = {}
        if "number_of_stories" in summary_measurements:
            house_data["number_of_stories"] = summary_measurements["number_of_stories"]
        if "total_roof_facets" in summary_measurements:
            facets = int(summary_measurements["total_roof_facets"])
            house_data["total_roof_facets"] = facets
            if facets < 7:
                house_data["structure_complexity"] = "Simple"
            elif facets <= 50:
                house_data["structure_complexity"] = "Normal"
            else:
                house_data["structure_complexity"] = "Complex"
        if "estimated_attic" in summary_measurements:
            house_data["estimated_attic"] = summary_measurements["estimated_attic"]
        if "total_roof_obstructions" in summary_measurements:
            house_data["total_roof_obstructions"] = int(summary_measurements["total_roof_obstructions"])

        if house_data:
            _add("C001", property_id, "House Measurements", "text", house_data)

    if summary_measurements or detailed_measurements:
        roof_data = {}
        if "total_roof_area" in summary_measurements:
            roof_data["total_area"] = summary_measurements["total_roof_area"]
        if "total_roof_facets" in summary_measurements:
            roof_data["total_roof_facets"] = int(summary_measurements["total_roof_facets"])
        if "predominant_pitch" in summary_measurements:
            roof_data["predominant_pitch"] = summary_measurements["predominant_pitch"]
        elif "predominant_pitch" in detailed_measurements:
            roof_data["predominant_pitch"] = detailed_measurements["predominant_pitch"]

        if "ridges" in detailed_measurements:
            roof_data["ridges"] = detailed_measurements["ridges"]
        if "hips" in detailed_measurements:
            roof_data["hips"] = detailed_measurements["hips"]
        if "valleys" in detailed_measurements:
            roof_data["valleys"] = detailed_measurements["valleys"]
        if "rakes" in detailed_measurements:
            roof_data["rakes"] = detailed_measurements["rakes"]
        if "eaves_starter" in detailed_measurements:
            roof_data["eaves_starters"] = detailed_measurements["eaves_starter"]
        if "flashing" in detailed_measurements:
            roof_data["flashing"] = detailed_measurements["flashing"]
        if "step_flashing" in detailed_measurements:
            roof_data["step_flashing"] = detailed_measurements["step_flashing"]
        if "parapet_walls" in detailed_measurements:
            roof_data["parapet_walls"] = detailed_measurements["parapet_walls"]

        if "roof_obstructions_perimeter" in summary_measurements:
            roof_data["roof_obstructions_perimeter"] = summary_measurements["roof_obstructions_perimeter"]
        if "roof_obstructions_area" in summary_measurements:
            roof_data["roof_obstructions_area"] = summary_measurements["roof_obstructions_area"]

        net_area = float(summary_measurements.get("total_roof_area", "0 sq ft").replace(" sq ft", "").replace(",", ""))
        obstruction_area = float(summary_measurements.get("roof_obstructions_area", "0 sq ft").replace(" sq ft", "").replace(",", ""))
        roof_data["net_roof_area"] = f"{net_area - obstruction_area:.1f} sq ft"

        if roof_data:
            _add("C002", property_id, "Roof Measurements", "text", roof_data)

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
