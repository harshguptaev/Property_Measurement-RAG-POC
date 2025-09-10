import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# -------------------- New Helper Parsers For Flattened Format (defined early for usage) --------------------
def _extract_measurements_structured(text: str) -> Dict[str, Any]:
    import re
    import html
    
    # Look for the structured measurements section - try both formats
    measurements_section_match = re.search(
        r'## Measurements\s*([\s\S]*?)(?=## Prepared For|$)', 
        text, 
        re.IGNORECASE
    )
    
    if not measurements_section_match:
        # Try without ## prefix (for docling format)
        measurements_section_match = re.search(
            r'Measurements\s*([\s\S]*?)(?=Prepared For|$)', 
            text, 
            re.IGNORECASE
        )
    
    data = {}
    
    if measurements_section_match:
        measurements_text = measurements_section_match.group(1)
        
        # The format has all labels first, then all values
        # Split on newlines and clean up
        lines = measurements_text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Separate labels (end with :) from values
        labels = []
        values = []
        
        for line in non_empty_lines:
            if line.endswith(':'):
                labels.append(line[:-1].strip().lower())  # Remove colon and normalize
            else:
                # Decode HTML entities like &lt; -> <
                decoded_value = html.unescape(line.strip())
                values.append(decoded_value)
        
        # Match labels with values
        # The values should be in the same order as the labels
        for i, label in enumerate(labels):
            if i < len(values):
                value = values[i]
                
                # Map labels to our keys
                if label == 'area':
                    data['area'] = value
                elif label == 'roof facets':
                    data['roof_facets'] = value
                elif label == 'predominant pitch':
                    data['predominant_pitch'] = value
                elif label == 'number of stories':
                    data['number_of_stories'] = value
                elif label == 'ridges/hips':
                    data['ridges_hips'] = value
                elif label == 'valleys':
                    data['valleys'] = value
                elif label == 'rakes':
                    data['rakes'] = value
                elif label == 'eaves':
                    data['eaves'] = value
                elif label == 'estimated attic':
                    data['estimated_attic'] = value
                elif label == 'roof penetrations':
                    data['roof_penetrations'] = value
                elif label == 'roof penetrations perimeter':
                    data['roof_penetrations_perimeter'] = value
                elif label == 'roof penetrations area':
                    data['roof_penetrations_area'] = value
    
    return data

def _extract_prepared_for_structured(text: str) -> Optional[Dict[str, Any]]:
    import re
    match = re.search(r"Prepared For\s*([\s\S]{0,200})", text, re.IGNORECASE)
    if not match:
        return None
    segment = match.group(0).split('\n')[:5]
    lines = [l.strip() for l in segment if l.strip() and not l.lower().startswith('open in eagleview')]
    if not lines:
        return None
    name = None
    phone = None
    phone_re = re.compile(r"(\(\d{3}\)\s*\d{3}-?\d{4})")
    cleaned_lines = []
    for l in lines:
        ph = phone_re.search(l)
        if ph:
            phone = ph.group(1)
            l = phone_re.sub("", l).strip()
        if not name:
            name = l
        else:
            cleaned_lines.append(l)
    return {"name": name, "address": ", ".join(cleaned_lines), "phone": phone}

def _extract_property_details_structured(text: str) -> Dict[str, Any]:
    import re
    patterns = {
        "total_roof_facets": r"Total Roof Facets\s*=\s*(\d+)",
        "total_roof_obstructions": r"Total Roof (?:Penetrations|Obstructions)\s*=\s*(\d+)",
        "ridges": r"Ridges\s*=\s*([0-9' \"()A-Za-z]+)",
        "hips": r"Hips\s*=\s*([0-9' \"()A-Za-z]+)",
        "valleys": r"Valleys\s*=\s*([0-9' \"()A-Za-z]+)",
        "rakes": r"Rakes[†]?\s*=\s*([0-9' \"()A-Za-z]+)",
        "eaves_starters": r"Eaves/Starters[‡]?\s*=\s*([0-9' \"()A-Za-z]+)",
        "drip_edge": r"Drip Edge .*?=\s*([0-9' \"()A-Za-z]+)",
        "parapet_walls": r"Parapet Walls\s*=\s*([0-9' \"()A-Za-z]+)",
        "flashing": r"Flashing\s*=\s*([0-9' \"()A-Za-z]+)",
        "step_flashing": r"Step Flashing\s*=\s*([0-9' \"()A-Za-z]+)",
        "total_roof_obstructions_area": r"Total Roof (?:Penetrations|Obstructions) Area\s*=\s*([0-9\.]+\s*SQ)",
        "total_roof_area_less_obstructions": r"Total Roof Area Less Roof (?:Penetrations|Obstructions)\s*=\s*([0-9\.]+\s*SQ)",
        "total_roof_obstructions_perimeter": r"Total Roof (?:Penetrations|Obstructions) Perimeter\s*=\s*([0-9' \"()A-Za-z]+)",
        "predominant_pitch": r"Predominant Pitch\s*=\s*([0-9/]+)",
        "total_area_all_pitches": r"Total Area \(All Pitches\)\s*=\s*([0-9\.]+\s*SQ)"
    }
    data = {}
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            data[key] = m.group(1).strip()
    return data

def _extract_legal_notice(text: str, report_id: str, property_address: str, date_val: str) -> Optional[Dict[str, Any]]:
    import re
    m = re.search(r"(IMPORTANT LEGAL NOTICE AND DISCLAIMER[\s\S]{0,1500})", text, re.IGNORECASE)
    if not m:
        m = re.search(r"(Legal Notice and Disclaimer[\s\S]{0,1500})", text, re.IGNORECASE)
    if not m:
        return None
    snippet = re.sub(r"\s+", " ", m.group(1)).strip()
    return {
        "report_id": report_id,
        "property_address": property_address,
        "date": date_val,
        "excerpt": snippet[:1000]
    }


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


def load_docling_export(pdf_path: str) -> Optional[Dict[str, Any]]:
    """Load Docling JSON export if available."""
    try:
        stem = Path(pdf_path).stem
        docling_json_path = Path("docling_exports") / stem / f"{stem}.json"
        if docling_json_path.exists():
            with open(docling_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load Docling export: {e}")
    return None


def normalize_lines(text: str) -> List[str]:
    raw_lines = text.splitlines()
    lines = [l.strip() for l in raw_lines]
    return [l for l in lines if l]


# Keep some of the original extraction functions for fallback
def extract_section_lines(all_pages_text: List[str], header: str, stop_headers: List[str]) -> List[str]:
    header_lower = header.lower()
    stop_set = {s.lower() for s in stop_headers}
    for page_text in all_pages_text:
        lines = normalize_lines(page_text)
        for idx, line in enumerate(lines):
            if line.lower() == header_lower:
                collected = []
                for j in range(idx + 1, len(lines)):
                    l = lines[j]
                    if l.lower() in stop_set:
                        break
                    if l.lower() == header_lower:
                        break
                    collected.append(l)
                if collected:
                    return collected
    return []


def extract_important_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract important chunks in the NEW flattened JSON format.

    Output Example (list):
    [
      {
        "chunk_id": "<report>_chunk_1",
        "section": "Report Header",
        "type": "text",
        "data": { ... }
      }, ...]
    """

    # Extract report ID from filename
    if 'RoofReport-' in Path(pdf_path).name:
        try:
            report_id = Path(pdf_path).name.split('RoofReport-')[1].split('.')[0]
        except Exception:
            report_id = Path(pdf_path).stem
    else:
        report_id = Path(pdf_path).stem

    docling_data = load_docling_export(pdf_path)

    # Combine all text (best-effort) for regex parsing
    all_text = ""
    if docling_data and "texts" in docling_data:
        all_text = " \n".join([t.get("text", "") for t in docling_data["texts"]])
    else:
        # Fallback to PyMuPDF text extraction
        try:
            pages = read_pdf_text_by_page(pdf_path)
            all_text = "\n".join(pages)
        except Exception:
            all_text = ""

    chunks: List[Dict[str, Any]] = []

    def _add(section: str, _type: str, data: Dict[str, Any]):
        chunk_id = f"{report_id}_chunk_{len(chunks)+1}"
        chunks.append({
            "chunk_id": chunk_id,
            "section": section,
            "type": _type,  # one of text|image|table
            "data": data
        })

    # ---------------- Report Header ----------------
    date_match = re.search(r'([A-Z][a-z]+\s+\d{1,2},\s*20\d{2})', all_text)
    date_val = date_match.group(1) if date_match else ""
    # Address: look for first occurrence of pattern with city, state zip
    # Prefer a clean address line: number street city, ST ZIP
    addr_match = re.search(r'(\d+\s+[A-Za-z0-9 .]+?,\s*[A-Za-z .]+?,?\s*[A-Z]{2}\s*\d{5})', all_text)
    property_address = ""
    if addr_match:
        property_address = re.sub(r'\s+', ' ', addr_match.group(1)).strip()
    _add("Report Header", "text", {
        "report_id": report_id,
        "date": date_val,
        "property_address": property_address
    })

    # ---------------- Measurements ----------------
    measurements_data = _extract_measurements_structured(all_text)
    if measurements_data:
        _add("Measurements", "text", measurements_data)

    # ---------------- Prepared For ----------------
    prepared_for = _extract_prepared_for_structured(all_text)
    if prepared_for:
        _add("Prepared For", "text", prepared_for)

    # ---------------- Diagrams (Images) ----------------
    # Map section name -> (expected filename, placeholder, description)
    image_mappings = [
        ("Lengths Diagram", "Lengthsimage.png", "[IMAGE_LENGTHS_DIAGRAM]", "Diagram showing roof length measurements including ridges, hips, valleys, rakes, eaves, flashing, step flashing, and parapets"),
        ("Pitch (Degrees) Diagram", "Pitch_Degrees.png", "[IMAGE_PITCH_DEGREES_DIAGRAM]", "Diagram showing roof pitch in degrees for different facets"),
        ("Pitch (on 12) Diagram", "Pitch_on_12.png", "[IMAGE_PITCH_ON_12_DIAGRAM]", "Diagram showing roof pitch in x/12 format for different facets"),
        ("Rafters Diagram", "Rafters.png", "[IMAGE_RAFTERS_DIAGRAM]", "Diagram showing rafter lengths for different roof sections"),
        ("Azimuth Diagram", "Azimuth.png", "[IMAGE_AZIMUTH_DIAGRAM]", "Diagram showing roof facet orientations in degrees relative to true north"),
        ("Area Diagram", "Area.png", "[IMAGE_AREA_DIAGRAM]", "Diagram showing roof area measurements for different facets in square feet"),
        ("Roof Obstructions Diagram", "Roof_Penetrations.png", "[IMAGE_ROOF_OBSTRUCTIONS_DIAGRAM]", "Diagram showing locations of roof obstructions")
    ]
    images_dir = Path("extracted_images") / f"report_{report_id}"
    for section, filename, placeholder, desc in image_mappings:
        if (images_dir / filename).exists():
            _add(section, "image", {"description": desc, "image_file": str(images_dir / filename)})

    # Property Imagery (aggregate) – use Top_View as representative if exists
    property_images = ["Top_View.png", "North_Side.png", "South_Side.png", "East_Side.png", "West_Side.png"]
    have_any = any((images_dir / f).exists() for f in property_images)
    if have_any:
        _add("Property Imagery", "image", {
            "description": "Aerial images of the property from top, north, south, east, and west views",
            "images": [str(images_dir / f) for f in property_images if (images_dir / f).exists()]
        })

    # ---------------- Roofing Report Summary (table placeholder) ----------------
    _add("Roofing Report Summary", "table", {})  # table intentionally left empty

    # ---------------- Property Details ----------------
    prop_details = _extract_property_details_structured(all_text)
    if prop_details:
        _add("Property Details", "text", prop_details)

    # ---------------- Longitude and Latitude (separate chunks) ----------------
    longitude_match = re.search(r"Longitude\s*=\s*([-0-9\.]+)", all_text, re.IGNORECASE)
    if longitude_match:
        _add("Longitude", "text", {
            "longitude": longitude_match.group(1).strip(),
            "coordinate_type": "longitude"
        })
    
    latitude_match = re.search(r"Latitude\s*=\s*([-0-9\.]+)", all_text, re.IGNORECASE)
    if latitude_match:
        _add("Latitude", "text", {
            "latitude": latitude_match.group(1).strip(),
            "coordinate_type": "latitude"
        })

    # ---------------- Legal Notice and Disclaimer ----------------
    legal = _extract_legal_notice(all_text, report_id, property_address, date_val)
    return chunks


def extract_from_docling(docling_data: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
    """(Legacy) kept for backward compatibility if needed elsewhere (no longer used)."""
    return {}


def get_image_name_by_index(index: int) -> str:
    """Map image index to specific image names based on the extracted_images structure."""
    image_names = [
        "Area", "Azimuth", "Cover_Image", "Cover_Image_2", "Cover_Image_3",
        "East_Side", "Lengthsimage", "North_Side", "Pitch_Degrees", "Pitch_on_12",
        "Rafters", "Roof_Penetrations", "South_Side", "Structure_Summary", "Top_View", "West_Side"
    ]
    if index < len(image_names):
        return image_names[index]
    return f"image_{index + 1}"


def extract_report_and_property_info(text: str) -> Optional[Dict[str, Any]]:
    """Extract Report ID and Property Address information."""
    # Look for report ID pattern
    report_match = re.search(r'REPORT ID[:\s]*(\d+)', text, re.IGNORECASE)
    date_match = re.search(r'(August \d+, \d+)', text)
    
    # Look for property address pattern - more flexible
    property_patterns = [
        r'PROPERTY[:\s]*([^\n]*(?:\n[^\n]*)*?)(?=\nPrepared|\n\n|\nLongitude)',
        r'Property Address[:\s]*([^\n]*(?:\n[^\n]*)*?)(?=\nReport|\n\n)',
        r'(\d+\s+[A-Za-z\s]+(?:Rd|Road|St|Street|Ave|Avenue|Dr|Drive|Blvd|Boulevard|Ln|Lane|Ct|Court|Way|Pl|Place)[^\n]*(?:\n[^\n]*)*?CT\s+\d{5})',
    ]
    
    property_text = None
    for pattern in property_patterns:
        property_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if property_match:
            property_text = property_match.group(1).strip()
            break
    
    # Also try to find the specific address format
    if not property_text:
        address_match = re.search(r'185 Sand Dam Rd[^\n]*Thompson[^\n]*CT[^\n]*06277', text, re.IGNORECASE)
        if address_match:
            property_text = address_match.group(0)
    
    # Combine the parts
    content_parts = []
    if report_match:
        content_parts.append(f"REPORT ID {report_match.group(1)}")
    if date_match:
        content_parts.append(date_match.group(1))
    
    if property_text:
        content_parts.append(f"PROPERTY\n{property_text}")
    
    if content_parts:
        return {
            "type": "report_and_property",
            "raw_text": "\n".join(content_parts),
            "src_image_path": ""
        }
    
    return None


def extract_prepared_for_chunk(text: str) -> Optional[Dict[str, Any]]:
    """Extract Prepared For information as a single chunk."""
    # More specific pattern for Prepared For section
    patterns = [
        r'(Prepared For[:\s]*\n?[^\n]*Divya Devadas[^\n]*(?:\n[^\n]*)*?98004[^\n]*(?:\n[^\n]*)*?\(\d{3}\)\s*\d{3}-?\d{4})',
        r'(Prepared For[:\s]*[^\n]*(?:\n[^\n]*)*?)(?=\nProperty Location|\nLongitude|\n\n|\nROOFING)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Clean up the content
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            # Filter out unwanted lines
            clean_lines = []
            for line in lines:
                if not any(skip in line.lower() for skip in ['open in eagleview', 'satisfaction guaranteed', 'www.eagleview.com']):
                    clean_lines.append(line)
            
            if clean_lines:
                return {
                    "type": "prepared_for",
                    "raw_text": "\n".join(clean_lines),
                    "src_image_path": ""
                }
    
    return None


def extract_measurements_chunk(text: str) -> Optional[Dict[str, Any]]:
    """Extract Lengths, Areas and Pitches information."""
    # Look for the measurements section with actual values from the PDF
    measurements_pattern = r'Lengths, Areas and Pitches[:\s]*[\s\S]*?(?=Property Location|Longitude|$)'
    section_match = re.search(measurements_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if section_match:
        section_text = section_match.group(0)
        
        # Extract individual measurements using regex patterns
        measurements_list = ['Lengths, Areas and Pitches']
        
        measurement_patterns = [
            (r'Ridges\s*=\s*([^(]*\([^)]*\))', 'Ridges'),
            (r'Hips\s*=\s*([^(]*\([^)]*\))', 'Hips'),
            (r'Valleys\s*=\s*([^(]*\([^)]*\))', 'Valleys'),
            (r'Rakes[†]?\s*=\s*([^(]*\([^)]*\))', 'Rakes†'),
            (r'Eaves[/\\]?Starters[‡]?\s*=\s*([^(]*\([^)]*\))', 'Eaves/Starters‡'),
            (r'Drip Edge[^=]*=\s*([^(]*\([^)]*\))', 'Drip Edge (Eaves + Rakes)'),
            (r'Parapet Walls\s*=\s*([^(]*\([^)]*\))', 'Parapet Walls'),
            (r'Flashing\s*=\s*([^(]*\([^)]*\))', 'Flashing'),
            (r'Step Flashing\s*=\s*([^(]*\([^)]*\))', 'Step Flashing'),
            (r'Total Roof Penetrations Area\s*=\s*([\d.]+\s*SQ)', 'Total Roof Penetrations Area'),
            (r'Total Roof Area Less Roof Penetrations\s*=?\s*([\d.]+\s*SQ)', 'Total Roof Area Less Roof Penetrations'),
            (r'Total Roof Penetrations Perimeter\s*=\s*([^\n]*?)(?=\s|$)', 'Total Roof Penetrations Perimeter'),
            (r'Predominant Pitch\s*=\s*([^\s]*)', 'Predominant Pitch'),
            (r'Total Area \(All Pitches\)\s*=\s*([\d.]+\s*SQ)', 'Total Area (All Pitches)')
        ]
        
        for pattern, label in measurement_patterns:
            match = re.search(pattern, section_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Clean up the value
                value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
                measurements_list.append(f"{label} = {value}")
        
        if len(measurements_list) > 1:  # More than just the header
            return {
                "type": "measurements",
                "raw_text": "\n".join(measurements_list),
                "src_image_path": ""
            }
    
    return None


def extract_property_location_chunk(text: str) -> Optional[Dict[str, Any]]:
    """Extract Property Location information."""
    pattern = r'(Property Location[:\s]*(?:\n[^\n]*)*?)(?=\n\n|\n[A-Z][A-Z])'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        content = match.group(1).strip()
        return {
            "type": "property_location",
            "raw_text": content,
            "src_image_path": ""
        }
    
    # Fallback: look for longitude and latitude
    lon_lat_pattern = r'(Longitude\s*=\s*[-\d.]+\s*Latitude\s*=\s*[-\d.]+)'
    fallback_match = re.search(lon_lat_pattern, text, re.IGNORECASE)
    
    if fallback_match:
        content = f"Property Location\n{fallback_match.group(1).strip()}"
        return {
            "type": "property_location",
            "raw_text": content,
            "src_image_path": ""
        }
    
    return None


def extract_structured_data_from_docling(docling_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract structured data like address and prepared_for from Docling data."""
    extracted = []
    
    # Combine all text conte nt for pattern matching
    all_text = ""
    if "texts" in docling_data:
        all_text = " ".join([text_elem.get("text", "") for text_elem in docling_data["texts"]])
    
    # Extract address
    address_data = extract_address_from_text(all_text)
    if address_data:
        extracted.append(address_data)
    
    # Extract prepared_for
    prepared_for_data = extract_prepared_for_from_text(all_text)
    if prepared_for_data:
        extracted.append(prepared_for_data)
    
    return extracted


def extract_address_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract address information from text."""
    # Look for address patterns
    address_patterns = [
        r'(\d+\s+[A-Za-z\s]+(?:Rd|Road|St|Street|Ave|Avenue|Dr|Drive|Blvd|Boulevard|Ln|Lane|Ct|Court|Way|Pl|Place)[\s,]*[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5})',
        r'Property Address[:\s]*([^\n]+)',
    ]
    
    for pattern in address_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            address_text = match.group(1).strip()
            # Split address into lines
            address_parts = [part.strip() for part in address_text.split(',')]
            
            return {
                "type": "address",
                "data": {
                    "lines": address_parts,
                    "single_line": address_text
                }
            }
    
    return None


def extract_prepared_for_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract prepared_for information from text."""
    # Look for prepared for section
    prepared_for_match = re.search(r'Prepared For\s*\n([^:]+?)(?=\n\n|\nProperty|\nInsurance|\n[A-Z])', text, re.IGNORECASE | re.DOTALL)
    
    if prepared_for_match:
        prepared_for_text = prepared_for_match.group(1).strip()
        lines = [line.strip() for line in prepared_for_text.split('\n') if line.strip()]
        
        # Extract name (first line)
        name = lines[0] if lines else None
        
        # Extract phone number
        phone_pattern = r'(\(\d{3}\)\s*\d{3}-?\d{4}|\d{3}[-\.\s]\d{3}[-\.\s]\d{4})'
        phone = None
        for line in lines:
            phone_match = re.search(phone_pattern, line)
            if phone_match:
                phone = phone_match.group(1)
                break
        
        # Filter out phone and keep address lines
        address_lines = []
        for line in lines[1:]:
            if not phone or phone not in line:
                # Skip lines that look like legal disclaimers
                if not any(skip in line.lower() for skip in ["open in eagleview", "satisfaction guaranteed", "www.eagleview.com"]):
                    address_lines.append(line)
        
        return {
            "type": "prepared_for",
            "data": {
                "name": name,
                "lines": address_lines,
                "single_line": ", ".join(address_lines),
                "phone": phone
            }
        }
    
    return None


def extract_from_text(pdf_path: str, report_id: str) -> Dict[str, Any]:
    """(Legacy) no longer used with new flattened format."""
    return {}


def split_text_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into reasonable chunks."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def write_chunks_output(pdf_path: str, chunks: List[Dict[str, Any]]) -> Path:
    stem = Path(pdf_path).stem
    out_dir = Path("docling_exports") / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "important_chunks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()
    pdf_path = os.path.abspath(args.pdf)
    chunks = extract_important_chunks(pdf_path)
    out_file = write_chunks_output(pdf_path, chunks)
    if args.print:
        print(json.dumps(chunks, ensure_ascii=False, indent=2))
    print(str(out_file))


if __name__ == "__main__":
    main()


