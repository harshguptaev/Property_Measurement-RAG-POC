import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict


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


def normalize_lines(text: str) -> List[str]:
    raw_lines = text.splitlines()
    lines = [l.strip() for l in raw_lines]
    return [l for l in lines if l]


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


def extract_property_address(all_pages_text: List[str]) -> Dict:
    lines = extract_section_lines(
        all_pages_text,
        header="Property Address",
        stop_headers=["Prepared For", "Insurance Carrier", "Policy", "Claim", "Inspection Date", "Lengths", "Measurements"],
    )
    if not lines:
        return {}
    
    # Filter out lines that look like measurements or other non-address content
    address_lines = []
    for line in lines:
        line_lower = line.lower()
        # Skip lines that are clearly not address components
        if any(skip_word in line_lower for skip_word in [
            "measurements", "area:", "roof facets:", "predominant pitch:", 
            "number of stories:", "ridges/hips:", "valleys:", "rakes:", 
            "eaves:", "estimated attic:", "roof penetrations:", "sq", "'"
        ]):
            break
        address_lines.append(line)
    
    if not address_lines:
        address_lines = lines[:3]  # Take first 3 lines as fallback
    
    return {
        "type": "address",
        "data": {
            "lines": address_lines,
            "single_line": ", ".join(address_lines),
        },
    }


def split_name_and_phone(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    phone_pattern = re.compile(
        r"""
        (\(\d{3}\)\s*\d{3}-?\d{4})|
        (\+?1?\s*\d{3}[-\.\s]\d{3}[-\.\s]\d{4})|
        (\d{3}[-\.\s]\d{3}[-\.\s]\d{4})
        """,
        re.X,
    )
    name = None
    phone = None
    if lines:
        name = lines[0]
    for l in lines:
        m = phone_pattern.search(l)
        if m:
            phone = m.group(0)
            break
    return name, phone


def extract_prepared_for(all_pages_text: List[str]) -> Dict:
    lines = extract_section_lines(
        all_pages_text,
        header="Prepared For",
        stop_headers=["Property Address", "Insurance Carrier", "Policy", "Claim", "Inspection Date", "Lengths"],
    )
    if not lines:
        return {}
    
    # Filter out legal/disclaimer text
    filtered_lines = []
    for line in lines:
        line_lower = line.lower()
        # Skip lines that are clearly legal disclaimers or system text
        if any(skip_phrase in line_lower for skip_phrase in [
            "open in eagleview", "satisfaction guaranteed", "www.eagleview.com",
            "this document is provided", "eagleview technologies", "terms and conditions",
            "internal use only", "subject to the terms", "prohibited", "requestor"
        ]):
            break
        filtered_lines.append(line)
    
    if not filtered_lines:
        filtered_lines = lines[:4]  # Take first 4 lines as fallback
    
    name, phone = split_name_and_phone(filtered_lines)
    address_lines = [l for l in filtered_lines[1:] if l != phone]
    
    return {
        "type": "prepared_for",
        "data": {
            "name": name,
            "lines": address_lines if address_lines else filtered_lines[1:],
            "phone": phone,
            "single_line": ", ".join(address_lines if address_lines else filtered_lines[1:]),
        },
    }


def parse_lengths_block(text: str) -> Dict:
    lines = normalize_lines(text)
    if not lines:
        return {}
    try:
        start_idx = None
        for i, l in enumerate(lines):
            if l.lower().startswith("lengths"):
                start_idx = i
                break
        if start_idx is None:
            return {}
        block = lines[start_idx:]
        data: Dict[str, str] = {}
        simple_pairs = {
            "ridges": re.compile(r"ridges\s*=\s*([^\n]+)", re.I),
            "valleys": re.compile(r"valleys\s*=\s*([^\n]+)", re.I),
            "rakes": re.compile(r"rakes\s*\u2020?\s*=\s*([^\n]+)", re.I),
            "eaves": re.compile(r"eaves(?:/starters\u2021?)?\s*=\s*([^\n]+)", re.I),
            "hips": re.compile(r"hips\s*=\s*([^\n]+)", re.I),
            "parapets": re.compile(r"parapets?\s*=\s*([^\n]+)", re.I),
            "flashing": re.compile(r"flashing\s*=\s*([^\n]+)", re.I),
            "step_flashing": re.compile(r"step\s*flashing\s*=\s*([^\n]+)", re.I),
            "drip_edge": re.compile(r"drip\s*edge.*=\s*([^\n]+)", re.I),
        }
        totals_pairs = {
            "total_roof_penetrations_area": re.compile(r"total\s*roof\s*penetrations\s*area\s*=\s*([^\n]+)", re.I),
            "total_roof_area_less_penetrations": re.compile(r"total\s*roof\s*area\s*less\s*roof\s*penetrations\s*=\s*([^\n]+)", re.I),
            "total_roof_penetrations_perimeter": re.compile(r"total\s*roof\s*penetrations\s*perimeter\s*=\s*([^\n]+)", re.I),
            "predominant_pitch": re.compile(r"predominant\s*pitch\s*=\s*([^\n]+)", re.I),
            "total_area_all_pitches": re.compile(r"total\s*area\s*\(all\s*pitches\)\s*=\s*([^\n]+)", re.I),
        }
        joined = "\n".join(block)
        for k, pat in simple_pairs.items():
            m = pat.search(joined)
            if m:
                data[k] = m.group(1).strip()
        totals: Dict[str, str] = {}
        for k, pat in totals_pairs.items():
            m = pat.search(joined)
            if m:
                totals[k] = m.group(1).strip()
        if totals:
            data.update(totals)
        if data:
            return data
        return {}
    except Exception:
        return {}


def extract_lengths_from_last_page(all_pages_text: List[str]) -> Dict:
    if not all_pages_text:
        return {}
    last_text = all_pages_text[-1] or ""
    data = parse_lengths_block(last_text)
    if not data:
        joined = "\n".join(all_pages_text)
        data = parse_lengths_block(joined)
    if not data:
        return {}
    return {"type": "lengths", "data": data}


def extract_roof_materials(all_pages_text: List[str]) -> Dict:
    """Extract roof material information from the document."""
    materials_data = {}
    
    for page_text in all_pages_text:
        lines = normalize_lines(page_text)
        text_lower = page_text.lower()
        
        # Look for common roofing materials
        materials = []
        if any(term in text_lower for term in ["asphalt", "shingle", "composition"]):
            materials.append("Asphalt Shingles")
        if any(term in text_lower for term in ["metal", "steel", "aluminum"]):
            materials.append("Metal Roofing")
        if any(term in text_lower for term in ["tile", "clay", "concrete tile"]):
            materials.append("Tile")
        if any(term in text_lower for term in ["slate"]):
            materials.append("Slate")
        if any(term in text_lower for term in ["wood", "cedar", "shake"]):
            materials.append("Wood")
        if any(term in text_lower for term in ["membrane", "tpo", "epdm", "modified bitumen"]):
            materials.append("Membrane")
        
        if materials:
            materials_data["detected_materials"] = list(set(materials))
            break
    
    return {"type": "materials", "data": materials_data} if materials_data else {}


def extract_roof_condition(all_pages_text: List[str]) -> Dict:
    """Extract roof condition information from the document."""
    condition_data = {}
    
    for page_text in all_pages_text:
        text_lower = page_text.lower()
        
        # Look for condition indicators
        conditions = []
        issues = []
        
        if any(term in text_lower for term in ["excellent", "good condition", "well maintained"]):
            conditions.append("Good")
        if any(term in text_lower for term in ["fair", "moderate", "some wear"]):
            conditions.append("Fair")
        if any(term in text_lower for term in ["poor", "damaged", "needs replacement"]):
            conditions.append("Poor")
        
        # Look for specific issues
        if any(term in text_lower for term in ["missing shingle", "loose shingle", "damaged shingle"]):
            issues.append("Shingle Issues")
        if any(term in text_lower for term in ["leak", "water damage", "moisture"]):
            issues.append("Water Damage")
        if any(term in text_lower for term in ["flashing", "damaged flashing"]):
            issues.append("Flashing Issues")
        if any(term in text_lower for term in ["gutter", "damaged gutter", "clogged"]):
            issues.append("Gutter Issues")
        if any(term in text_lower for term in ["penetration", "roof penetration"]):
            issues.append("Penetrations")
        
        if conditions or issues:
            condition_data["overall_condition"] = conditions
            condition_data["identified_issues"] = issues
            break
    
    return {"type": "condition", "data": condition_data} if condition_data else {}


def extract_inspection_details(all_pages_text: List[str]) -> Dict:
    """Extract inspection date and other details."""
    inspection_data = {}
    
    for page_text in all_pages_text:
        lines = normalize_lines(page_text)
        
        for line in lines:
            line_lower = line.lower()
            
            # Look for inspection date
            if "inspection date" in line_lower:
                # Try to extract date from the line or next few lines
                date_match = re.search(r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', line)
                if date_match:
                    inspection_data["inspection_date"] = date_match.group(1)
            
            # Look for report ID
            if "report id" in line_lower or "report number" in line_lower:
                # Extract ID from the line
                id_match = re.search(r'(\d+)', line)
                if id_match:
                    inspection_data["report_id"] = id_match.group(1)
            
            # Look for number of stories
            if "number of stories" in line_lower or "stories" in line_lower:
                stories_match = re.search(r'(\d+)', line)
                if stories_match:
                    inspection_data["number_of_stories"] = stories_match.group(1)
    
    return {"type": "inspection_details", "data": inspection_data} if inspection_data else {}


def extract_roof_geometry(all_pages_text: List[str]) -> Dict:
    """Extract detailed roof geometry information."""
    geometry_data = {}
    
    for page_text in all_pages_text:
        lines = normalize_lines(page_text)
        text_lower = page_text.lower()
        
        # Look for roof facets
        facets_match = re.search(r'roof facets[:\s]*(\d+)', text_lower)
        if facets_match:
            geometry_data["roof_facets"] = facets_match.group(1)
        
        # Look for roof complexity indicators
        if any(term in text_lower for term in ["complex", "multiple levels", "irregular"]):
            geometry_data["complexity"] = "Complex"
        elif any(term in text_lower for term in ["simple", "single level", "regular"]):
            geometry_data["complexity"] = "Simple"
        
        # Extract pitch variations
        pitch_matches = re.findall(r'(\d+/\d+)', page_text)
        if pitch_matches:
            geometry_data["pitch_variations"] = list(set(pitch_matches))
    
    return {"type": "geometry", "data": geometry_data} if geometry_data else {}


def extract_important_chunks(pdf_path: str) -> List[Dict]:
    pages = read_pdf_text_by_page(pdf_path)
    chunks: List[Dict] = []
    
    # Extract existing chunks
    address = extract_property_address(pages)
    if address:
        chunks.append(address)
    
    prepared = extract_prepared_for(pages)
    if prepared:
        chunks.append(prepared)
    
    lengths = extract_lengths_from_last_page(pages)
    if lengths:
        chunks.append(lengths)
    
    # Extract new enhanced chunks
    materials = extract_roof_materials(pages)
    if materials:
        chunks.append(materials)
    
    condition = extract_roof_condition(pages)
    if condition:
        chunks.append(condition)
    
    inspection = extract_inspection_details(pages)
    if inspection:
        chunks.append(inspection)
    
    geometry = extract_roof_geometry(pages)
    if geometry:
        chunks.append(geometry)
    
    return chunks


def write_chunks_output(pdf_path: str, chunks: List[Dict]) -> Path:
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


