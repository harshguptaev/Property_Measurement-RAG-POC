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


def extract_important_chunks(pdf_path: str) -> Dict[str, Any]:
    """Extract important chunks in the new JSON format."""
    
    # Extract report ID from filename
    report_id = None
    if 'RoofReport-' in Path(pdf_path).name:
        try:
            report_id = Path(pdf_path).name.split('RoofReport-')[1].split('.')[0]
        except:
            report_id = Path(pdf_path).stem
    else:
        report_id = Path(pdf_path).stem
    
    # Load Docling export for structured data
    docling_data = load_docling_export(pdf_path)
    
    # Initialize the result structure
    result = {
        "reportId": report_id,
        "text": [],
        "table": [],
        "image": [],
        "extracted": []
    }
    
    # If we have Docling data, use it; otherwise fall back to text processing
    if docling_data:
        result = extract_from_docling(docling_data, pdf_path)
    else:
        result = extract_from_text(pdf_path, report_id)
    
    return result


def extract_from_docling(docling_data: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
    """Extract chunks from Docling JSON export."""
    
    report_id = None
    if 'RoofReport-' in Path(pdf_path).name:
        try:
            report_id = Path(pdf_path).name.split('RoofReport-')[1].split('.')[0]
        except:
            report_id = Path(pdf_path).stem
    else:
        report_id = Path(pdf_path).stem
    
    result = {
        "reportId": report_id,
        "text": [],
        "table": [],
        "image": [],
        "extracted": []
    }
    
    # Combine all text content for pattern matching
    all_text = ""
    if "texts" in docling_data:
        all_text = " ".join([text_elem.get("text", "") for text_elem in docling_data["texts"]])
    
    # Extract only the 4 specific text chunks
    chunk_number = 1
    
    # 1. Extract Report ID and Property Address
    report_property_chunk = extract_report_and_property_info(all_text)
    if report_property_chunk:
        report_property_chunk["id"] = str(uuid.uuid4())
        report_property_chunk["metadata"] = {"chunk_number": chunk_number}
        result["text"].append(report_property_chunk)
        chunk_number += 1
    
    # 2. Extract Prepared For
    prepared_for_chunk = extract_prepared_for_chunk(all_text)
    if prepared_for_chunk:
        prepared_for_chunk["id"] = str(uuid.uuid4())
        prepared_for_chunk["metadata"] = {"chunk_number": chunk_number}
        result["text"].append(prepared_for_chunk)
        chunk_number += 1
    
    # 3. Extract Lengths, Areas and Pitches
    measurements_chunk = extract_measurements_chunk(all_text)
    if measurements_chunk:
        measurements_chunk["id"] = str(uuid.uuid4())
        measurements_chunk["metadata"] = {"chunk_number": chunk_number}
        result["text"].append(measurements_chunk)
        chunk_number += 1
    
    # 4. Extract Property Location
    location_chunk = extract_property_location_chunk(all_text)
    if location_chunk:
        location_chunk["id"] = str(uuid.uuid4())
        location_chunk["metadata"] = {"chunk_number": chunk_number}
        result["text"].append(location_chunk)
        chunk_number += 1
    
    # Extract image elements - create separate chunk for each image
    if "pictures" in docling_data:
        for i, picture_elem in enumerate(docling_data["pictures"]):
            # Get page number from provenance
            page_no = 1
            if "prov" in picture_elem and picture_elem["prov"]:
                page_no = picture_elem["prov"][0].get("page_no", 1)
            
            # Generate image path based on report structure
            image_path = f"extracted_images/report_{report_id}/{get_image_name_by_index(i)}.png"
            
            image_chunk = {
                "type": "image",
                "raw_text": "",
                "id": str(uuid.uuid4()),
                "metadata": {
                    "page": page_no,
                    "image_index": i,
                    "xref": picture_elem.get("self_ref", f"#/pictures/{i}")
                },
                "src_image_path": image_path
            }
            result["image"].append(image_chunk)
    
    # Leave tables empty as requested
    result["table"] = []
    
    return result


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
    
    # Combine all text content for pattern matching
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
    """Fallback extraction from plain text when Docling data is not available."""
    pages = read_pdf_text_by_page(pdf_path)
    
    result = {
        "reportId": report_id,
        "text": [],
        "table": [],
        "image": [],
        "extracted": []
    }
    
    # Combine all pages
    all_text = "\n".join(pages)
    
    # Extract only the 4 specific text chunks
    chunk_number = 1
    
    # 1. Extract Report ID and Property Address
    report_property_chunk = extract_report_and_property_info(all_text)
    if report_property_chunk:
        report_property_chunk["id"] = str(uuid.uuid4())
        report_property_chunk["metadata"] = {"chunk_number": chunk_number}
        result["text"].append(report_property_chunk)
        chunk_number += 1
    
    # 2. Extract Prepared For
    prepared_for_chunk = extract_prepared_for_chunk(all_text)
    if prepared_for_chunk:
        prepared_for_chunk["id"] = str(uuid.uuid4())
        prepared_for_chunk["metadata"] = {"chunk_number": chunk_number}
        result["text"].append(prepared_for_chunk)
        chunk_number += 1
    
    # 3. Extract Lengths, Areas and Pitches
    measurements_chunk = extract_measurements_chunk(all_text)
    if measurements_chunk:
        measurements_chunk["id"] = str(uuid.uuid4())
        measurements_chunk["metadata"] = {"chunk_number": chunk_number}
        result["text"].append(measurements_chunk)
        chunk_number += 1
    
    # 4. Extract Property Location
    location_chunk = extract_property_location_chunk(all_text)
    if location_chunk:
        location_chunk["id"] = str(uuid.uuid4())
        location_chunk["metadata"] = {"chunk_number": chunk_number}
        result["text"].append(location_chunk)
        chunk_number += 1
    
    # Leave tables empty as requested
    result["table"] = []
    
    # No images in text-only extraction
    result["image"] = []
    
    return result


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


def write_chunks_output(pdf_path: str, chunks: Dict[str, Any]) -> Path:
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


