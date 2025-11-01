import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from .property_rag_status_dao import PropertyRAGStatusDAO
from .db_connector import db_connector


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

    # Try to find address after "Property Address:" to get the clean address
    address_match = re.search(r'Property Address:\s*([^,\n]+,\s*[^,\n]+,\s*[A-Z]{2}\s+\d{5})', text, re.IGNORECASE)
    if not address_match:
        # Fallback: try "PROPERTY" keyword to avoid date prefixes
        address_match = re.search(r'PROPERTY\s+([^,\n]+,\s*[^,\n]+,\s*[A-Z]{2}\s+\d{5})', text, re.IGNORECASE)
    if not address_match:
        # Final fallback: look for any address pattern (original regex)
        address_match = re.search(r'(\d+[^,\n]+(?:,\s*[^,\n]+)?,\s*[A-Z]{2}\s+\d{5})', text)

    if address_match:
        # Clean up extra spaces in the address
        address = address_match.group(1).strip()
        # Remove any remaining date/property prefixes if they exist
        address = re.sub(r'^\d{1,2}/\d{1,2}/\d{4}\s+PROPERTY\s+', '', address, flags=re.IGNORECASE)
        address = re.sub(r'^\d{1,2},\s+\d{4}\s+PROPERTY\s+', '', address, flags=re.IGNORECASE)
        # Split on new address pattern (digit followed by space and letter) and take first address
        address_parts = re.split(r'\s+(?=\d+\s+[A-Za-z])', address)
        if address_parts:
            address = address_parts[0].strip()
        # Remove delimiter commas and normalize spacing
        address = re.sub(r',\s+', ' ', address)  # Remove commas used as delimiters
        address = re.sub(r'\s+', ' ', address)     # Normalize multiple spaces to single space
        data['property_address'] = address

    report_match = re.search(r'Report:\s*(\d+)', text)
    if report_match:
        data['report_id'] = report_match.group(1)

    return data


def _extract_prepared_for_premium(text: str) -> Optional[Dict[str, Any]]:
    """Extract prepared for information from premium report."""
    # Look for the line immediately following "## Prepared for"
    prepared_match = re.search(r'## Prepared for\s*\n\s*([^\n]+)', text, re.IGNORECASE)
    if not prepared_match:
        return None

    # The owner information is all in one line
    owner_line = prepared_match.group(1).strip()

    # Parse the owner line - typically: "Name Company Address Phone"
    # We'll take everything as contact for now
    if owner_line:
        data = {}
        data['contact'] = owner_line
        return data

    return None


def _extract_summary_measurements(text: str) -> Dict[str, Any]:
    """Extract summary measurements from the measurements section."""
    data = {}

    # First try the original patterns
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

    # If no data found, try the table format
    if not data:
        # Look for measurements in table format like | Area: | 3,721 sq ft |
        table_match = re.search(r'## Measurements\s*(.*?)(?=##|\n\n##)', text, re.DOTALL)
        if table_match:
            table_content = table_match.group(1)

            # Parse table rows
            lines = [line.strip() for line in table_content.split('\n') if line.strip() and '|' in line and not line.startswith('|--')]

            field_mapping = {
                'area:': 'total_roof_area',
                'roof facets:': 'total_roof_facets',
                'predominant pitch:': 'predominant_pitch',
                'number of stories:': 'number_of_stories',
                'ridges/hips:': 'total_ridges_hips',
                'valleys:': 'total_valleys',
                'rakes:': 'total_rakes',
                'eaves:': 'total_eaves',
                'estimated attic:': 'estimated_attic',
                'roof obstructions:': 'total_roof_obstructions',
                'roof obstructions perimeter:': 'roof_obstructions_perimeter',
                'roof obstructions area:': 'roof_obstructions_area'
            }

            for line in lines:
                if '|' in line:
                    parts = [part.strip() for part in line.split('|')[1:-1]]  # Skip empty parts at start/end
                    if len(parts) >= 2:
                        field_name = parts[0].lower().strip()
                        field_value = parts[1].strip()

                        # Map field name to key - check in order of specificity
                        # Check longer/more specific patterns first
                        matched = False
                        for pattern in ['roof obstructions area:', 'roof obstructions perimeter:', 'roof obstructions:', 'estimated attic:', 'number of stories:', 'predominant pitch:', 'roof facets:', 'ridges/hips:', 'valleys:', 'rakes:', 'eaves:', 'area:']:
                            if pattern in field_name:
                                data[field_mapping[pattern]] = field_value
                                matched = True
                                break

                        if not matched:
                            # Fallback to any match
                            for pattern, key in field_mapping.items():
                                if pattern in field_name:
                                    data[key] = field_value
                                    break

    return data


def _extract_detailed_measurements(text: str) -> Dict[str, Any]:
    """Extract detailed measurements from the lengths section, handling multiple structures."""
    structures_data = {}

    # Find all structure sections - handle both markdown headers and plain text
    structure_pattern = r'(?:## Structure #(\d+)|Structure\s+(\d+)|All Structures)'
    structure_matches = list(re.finditer(structure_pattern, text, re.IGNORECASE))

    if not structure_matches:
        # No structure sections found, extract as single structure
        structures_data['all'] = _extract_single_structure_measurements(text, "All Structures")
    else:
        # Extract measurements for each structure
        for i, match in enumerate(structure_matches):
            structure_name = match.group(0).strip()
            if structure_name.lower() == 'all structures':
                structure_key = 'all_structures'
            else:
                # Get structure number from either group 1 or 2 (markdown header or plain text)
                structure_num = match.group(1) or match.group(2)
                structure_key = f'structure_{structure_num}'

            # Extract text for this structure (from current match to next match or end)
            start_pos = match.end()
            end_pos = structure_matches[i + 1].start() if i + 1 < len(structure_matches) else len(text)

            structure_text = text[start_pos:end_pos]
            structures_data[structure_key] = _extract_single_structure_measurements(structure_text, structure_name)

    return structures_data


def _extract_single_structure_measurements(text: str, structure_name: str) -> Dict[str, Any]:
    """Extract measurements for a single structure."""
    data = {}

    patterns = {
        'ridges': r'Ridges\s*=\s*([0-9]+\.?\d*(?:\s*ft)?)\s*\(([0-9]+)\s*Ridges?\)',
        'hips': r'Hips\s*=\s*([0-9]+\.?\d*(?:\s*ft)?)\s*\(([0-9]+)\s*Hips?\)',
        'valleys': r'Valleys\s*=\s*([0-9]+\.?\d*(?:\s*ft)?)\s*\(([0-9]+)\s*Valleys?\)',
        'rakes': r'Rakes[^\w]*=\s*([0-9]+\.?\d*(?:\s*ft)?)\s*\(([0-9]+)\s*Rakes?\)',
        'eaves_starter': r'Eaves/Starter[^\w]*=\s*([0-9]+\.?\d*(?:\s*ft)?)\s*\(([0-9]+)\s*Eaves?\)',
        'flashing': r'Flashing\s*=\s*([0-9]+\.?\d*(?:\s*ft)?)\s*\(([0-9]+)\s*Lengths?\)',
        'step_flashing': r'Step flashing\s*=\s*([0-9]+\.?\d*(?:\s*ft)?)\s*\(([0-9]+)\s*Lengths?\)',
        'parapet_walls': r'Parapet Walls\s*=\s*([0-9]+\.?\d*(?:\s*ft)?)\s*\(([0-9]+)\s*Lengths?\)',
        'drip_edge': r'Drip Edge\s*\(Eaves\s*\+\s*Rakes\)\s*=\s*([0-9]+\.?\d*(?:\s*ft)?)\s*\(([0-9]+)\s*Lengths?\)',
        'predominant_pitch': r'Predominant Pitch\s*=\s*([0-9/°]+)',
        'total_area_all_pitches': r'Total Area\s*\(All Pitches\)\s*=\s*([0-9,]+\.?\d*\s*sq ft)',
        'net_roof_area': r'Total Roof Area Less Roof Obstructions\s*=\s*([0-9,]+\.?\d*\s*sq ft)',
        'roof_obstructions_area': r'Total Roof Obstructions Area\s*=\s*([0-9]+\.?\d*\s*sq ft)',
        'roof_obstructions_perimeter': r'Total Roof Obstructions Perimeter\s*=\s*([0-9]+\.?\d*\s*ft)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if len(match.groups()) == 2:
                # Patterns with count (measurement and count)
                measurement = match.group(1).strip()
                count = match.group(2).strip()
                # Add " ft" if not present
                if not measurement.endswith(' ft'):
                    measurement = f"{measurement} ft"
                if key == 'eaves_starter':
                    data[key] = f"{measurement} ({count} Eaves)"
                elif key == 'parapet_walls':
                    data[key] = f"{measurement} ({count} Lengths)"
                elif key in ['flashing', 'step_flashing', 'drip_edge']:
                    data[key] = f"{measurement} ({count} Lengths)"
                else:
                    data[key] = f"{measurement} ({count} {key.title()})"
            else:
                # Single group patterns
                data[key] = match.group(1).strip()

    return data


def _extract_pitch_breakdown(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Extract pitch breakdown table from report summary, handling multiple structures."""
    structures_pitch_data = {}

    # Find all structure sections for pitch breakdown - handle both markdown headers and plain text
    structure_pattern = r'(?:## Structure #(\d+)|Structure\s+(\d+)|All Structures)'
    structure_matches = list(re.finditer(structure_pattern, text, re.DOTALL | re.IGNORECASE))

    if not structure_matches:
        # No structure sections found, extract as single structure
        structures_pitch_data['all'] = _extract_single_pitch_breakdown(text, text)
    else:
        # Check if this is a single structure report (has All Structures but no individual structures)
        individual_structure_matches = [match for match in structure_matches if not match.group(0).lower().strip().startswith('all')]
        
        if not individual_structure_matches:
            # Single structure report - extract pitch data from the full text
            structures_pitch_data['all_structures'] = _extract_single_pitch_breakdown(text, text)
        else:
            # Multi-structure report - extract pitch data for each individual structure
            for i, match in enumerate(individual_structure_matches):
                structure_name = match.group(0).strip()
                # Get structure number from either group 1 or 2 (markdown header or plain text)
                structure_num = match.group(1) or match.group(2)
                structure_key = f'structure_{structure_num}'

                # Extract text for this structure (from current match to next match or end)
                start_pos = match.end()
                end_pos = individual_structure_matches[i + 1].start() if i + 1 < len(individual_structure_matches) else len(text)

                structure_text = text[start_pos:end_pos]
                structures_pitch_data[structure_key] = _extract_single_pitch_breakdown(structure_text, text)

    return structures_pitch_data


def _extract_single_pitch_breakdown(text: str, full_text: str = None) -> List[Dict[str, Any]]:
    """Extract pitch breakdown for a single structure."""
    pitch_data = []

    # Look for the pitch breakdown table - handle both formats
    table_match = re.search(r'\| Areas per Pitch.*?\|(.*?)(?=\n\n|\n##|\| Structure Complexity)', text, re.DOTALL)
    if table_match:
        table_content = table_match.group(1).strip()

        # Parse the table rows
        lines = [line.strip() for line in table_content.split('\n') if line.strip() and not line.startswith('|---')]

        pitches = []
        areas = []
        percentages = []

        for line in lines:
            if '|' in line:
                parts = [part.strip() for part in line.split('|')[1:-1]]  # Skip first and last empty parts
                if not parts:  # Skip empty lines
                    continue

                # Check row type based on content
                if 'Roof Pitches' in line:
                    # Pitch row - extract pitch values
                    pitches = [p for p in parts[1:] if p and '/' in p]  # Skip header, get pitches
                elif 'Area (sq ft)' in line:
                    # Area header row - extract area values
                    areas = [p for p in parts[1:] if p and any(c.isdigit() for c in p)]  # Skip header, get areas
                elif '%of Roof' in line:
                    # Percentage row - extract percentage values
                    percentages = [p for p in parts[1:] if p and '%' in p]  # Skip header, get percentages
                elif parts[0] == '' and any(p and any(c.isdigit() for c in p) for p in parts[1:]):
                    # Continuation area row (empty header, contains numbers) - add to areas
                    new_areas = [p for p in parts[1:] if p and any(c.isdigit() for c in p)]
                    areas.extend(new_areas)

        # Handle special case for Structure #2 where pitches are not in the table header
        # but we can infer them from the All Structures section
        if not pitches and areas and percentages and full_text:
            # Try to infer pitches by matching areas from All Structures section
            # Extract the entire All Structures table
            all_struct_table_match = re.search(r'## All Structures(.*?)(?=\n##|\n\nThe table above)', full_text, re.DOTALL)
            if all_struct_table_match:
                table_content = all_struct_table_match.group(1).strip()

                # Parse the table to get pitches and areas
                lines = [line.strip() for line in table_content.split('\n') if line.strip() and not line.startswith('|---')]

                all_pitches = []
                all_areas = []

                for line in lines:
                    if '|' in line:
                        parts = [part.strip() for part in line.split('|')[1:-1]]  # Skip first and last empty parts

                        if 'Roof Pitches' in line:
                            all_pitches = [p for p in parts if p and '/' in p]
                        elif 'Area (sq ft)' in line:
                            all_areas = [p for p in parts if p and any(c.isdigit() for c in p)]

                # Create a mapping from area to pitch
                area_to_pitch = {}
                for i in range(min(len(all_pitches), len(all_areas))):
                    try:
                        area_val = float(all_areas[i].replace(',', ''))
                        area_to_pitch[area_val] = all_pitches[i]
                    except (ValueError, IndexError):
                        continue

                # Match our areas to pitches
                pitches = []
                for area in areas:
                    if area:
                        try:
                            area_val = float(area.replace(',', ''))
                            # Find closest match
                            closest_pitch = ''
                            min_diff = float('inf')
                            for ref_area, pitch in area_to_pitch.items():
                                diff = abs(area_val - ref_area)
                                if diff < min_diff:
                                    min_diff = diff
                                    closest_pitch = pitch
                            pitches.append(closest_pitch)
                        except (ValueError, IndexError):
                            pitches.append('')
                    else:
                        pitches.append('')

        # If we still don't have pitches but have areas and percentages, create entries without pitches
        if not pitches and (areas or percentages):
            max_len = max(len(areas), len(percentages))
            for i in range(max_len):
                pitch_entry = {
                    'pitch': '',
                    'area_sq_ft': areas[i] if i < len(areas) else '',
                    'percentage': percentages[i] if i < len(percentages) else ''
                }
                pitch_data.append(pitch_entry)
        else:
            # Combine pitches, areas, and percentages
            max_len = max(len(pitches), len(areas), len(percentages))
            for i in range(max_len):
                pitch_entry = {
                    'pitch': pitches[i] if i < len(pitches) else '',
                    'area_sq_ft': areas[i] if i < len(areas) else '',
                    'percentage': percentages[i] if i < len(percentages) else ''
                }
                if pitch_entry['pitch'] or pitch_entry['area_sq_ft'] or pitch_entry['percentage']:
                    pitch_data.append(pitch_entry)

    # Fallback to original format if table parsing fails
    if not pitch_data:
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


def _extract_waste_calculation(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Extract waste calculation table, handling multiple structures."""
    structures_waste_data = {}

    # Find all structure sections for waste calculation - handle both markdown headers and plain text
    structure_pattern = r'(?:## Structure #(\d+)|Structure\s+(\d+)|All Structures)'
    structure_matches = list(re.finditer(structure_pattern, text, re.DOTALL | re.IGNORECASE))

    if not structure_matches:
        # No structure sections found, extract as single structure
        structures_waste_data['all'] = _extract_single_waste_calculation(text)
    else:
        # Check if this is a single structure report (has All Structures but no individual structures)
        individual_structure_matches = [match for match in structure_matches if not match.group(0).lower().strip().startswith('all')]
        
        if not individual_structure_matches:
            # Single structure report - extract waste data from the full text
            structures_waste_data['all_structures'] = _extract_single_waste_calculation(text)
        else:
            # Multi-structure report - extract waste data for each individual structure
            for i, match in enumerate(individual_structure_matches):
                structure_name = match.group(0).strip()
                # Get structure number from either group 1 or 2 (markdown header or plain text)
                structure_num = match.group(1) or match.group(2)
                structure_key = f'structure_{structure_num}'

                # Extract text for this structure (from current match to next match or end)
                start_pos = match.end()
                end_pos = individual_structure_matches[i + 1].start() if i + 1 < len(individual_structure_matches) else len(text)

                structure_text = text[start_pos:end_pos]
                structures_waste_data[structure_key] = _extract_single_waste_calculation(structure_text)

    return structures_waste_data


def _extract_single_waste_calculation(text: str) -> List[Dict[str, Any]]:
    """Extract waste calculation for a single structure."""
    waste_data = []

    # Look for the waste calculation table - handle markdown table format
    waste_match = re.search(r'\| Waste%.*?\|(.*?)(?=\n\n|\n##|\| Measured|\* Squares)', text, re.DOTALL)
    if waste_match:
        table_content = waste_match.group(1).strip()

        # Parse the table rows
        lines = [line.strip() for line in table_content.split('\n') if line.strip() and not line.startswith('|---')]

        percentages = []
        areas = []
        squares = []

        for line in lines:
            if '|' in line:
                parts = [part.strip() for part in line.split('|')[1:-1]]  # Skip first and last empty parts
                if not parts:  # Skip empty lines
                    continue

                # Check if this is a header row or data row
                if 'Area (Sq ft)' in line:
                    # This is the area row - extract all numeric values
                    areas = [p for p in parts[1:] if p and any(c.isdigit() for c in p)]  # Skip the header
                elif 'Squares' in line:
                    # This is the squares row - extract all values
                    squares = [p for p in parts[1:] if p]  # Skip the header
                elif '%' in line and not any(word in line for word in ['Area', 'Squares']):
                    # This is the percentage row - all parts are percentages
                    percentages = [p for p in parts if p and '%' in p]

        # Combine the data - they should all be the same length
        max_len = max(len(percentages), len(areas), len(squares))
        for i in range(max_len):
            waste_entry = {
                'waste_percentage': percentages[i] if i < len(percentages) else '',
                'area_sq_ft': areas[i] if i < len(areas) else '',
                'squares': squares[i] if i < len(squares) else ''
            }
            if waste_entry['waste_percentage'] or waste_entry['area_sq_ft'] or waste_entry['squares']:
                waste_data.append(waste_entry)

    # Fallback to original format if table parsing fails
    if not waste_data:
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


def _addresses_similar(address1: str, address2: str) -> bool:
    """
    Check if two addresses are similar (case-insensitive exact match for now).
    Can be enhanced with fuzzy matching later.
    """
    if not address1 or not address2:
        return False

    # Normalize addresses for comparison
    addr1_normalized = address1.lower().strip()
    addr2_normalized = address2.lower().strip()

    # Remove common separators and extra spaces
    addr1_normalized = re.sub(r'[,\s]+', ' ', addr1_normalized)
    addr2_normalized = re.sub(r'[,\s]+', ' ', addr2_normalized)

    # Exact match for now - can be enhanced with fuzzy matching
    return addr1_normalized == addr2_normalized


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

    # Extract address first to check for duplicates
    # Try to read from docling export first, fall back to PDF text
    temp_pages = []
    temp_text = ""
    if report_id_match:
        temp_report_id = report_id_match.group(1)
        docling_md_path = Path("docling_exports") / f"report_{temp_report_id}" / f"report_{temp_report_id}.md"
        if docling_md_path.exists():
            with open(docling_md_path, 'r', encoding='utf-8') as f:
                temp_text = f.read()
        else:
            temp_pages = read_pdf_text_by_page(pdf_path)
            temp_text = "\n".join(temp_pages)

    # Extract address to check for duplicates
    temp_header_data = _extract_premium_header(temp_text)
    extracted_address = temp_header_data.get('property_address', '').strip()

    if extracted_address:
        # Check if similar address already exists in database
        if db_connector.db_available:
            try:
                # Get all existing records to check for address similarity
                all_records = db_connector.execute_query("SELECT id, address FROM truedesigndemo.property_rag_status WHERE address IS NOT NULL")
                if all_records:
                    for record in all_records:
                        existing_address = record.get('address', '').strip()
                        if existing_address and _addresses_similar(extracted_address, existing_address):
                            print(f"Similar address already exists in database: '{existing_address}' matches '{extracted_address}'. Skipping processing.")
                            return []  # Return empty chunks to skip processing
            except Exception as db_error:
                print(f"Error checking for duplicate addresses: {db_error}")
                # Continue with processing if database check fails

    # If we get here, no duplicate address was found - proceed with processing
    # Create database record if address was extracted
    if extracted_address and db_connector.db_available:
        try:
            record_id = PropertyRAGStatusDAO.insert_record(
                address=extracted_address,
                report_id=report_id,
                product_id="13",
                lat=None,  # Will be updated later with coordinates
                lng=None   # Will be updated later with coordinates
            )
            if record_id != -1:
                print(f"Created database record with ID {record_id} for address: {extracted_address}")
            else:
                print("Database not available - proceeding without database tracking")
        except Exception as db_error:
            print(f"Error creating database record: {db_error}")

    # Try to read from docling export first, fall back to PDF text
    report_id_match = re.search(r'report_(\d+)', Path(pdf_path).stem)
    if report_id_match:
        report_id = report_id_match.group(1)
        docling_md_path = Path("docling_exports") / f"report_{report_id}" / f"report_{report_id}.md"
        if docling_md_path.exists():
            with open(docling_md_path, 'r', encoding='utf-8') as f:
                all_text = f.read()
        else:
            pages = read_pdf_text_by_page(pdf_path)
            all_text = "\n".join(pages)
    else:
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
    pitch_breakdown = _extract_pitch_breakdown(all_text)
    waste_calculation = _extract_waste_calculation(all_text)

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

        # Add number of structures
        num_structures = len([k for k in detailed_measurements.keys() if k.startswith('structure_')])
        if num_structures == 0:
            # Check if we have 'all' or 'all_structures' which indicates single structure
            if 'all' in detailed_measurements or 'all_structures' in detailed_measurements:
                num_structures = 1
        house_data["number_of_structures"] = num_structures

        if house_data:
            _add("C001", property_id, "House Measurements", "text", house_data)

    # Create roof measurement chunks for each structure
    structure_counter = 1
    for structure_key, measurements in detailed_measurements.items():
        if not measurements:  # Skip empty structures
            continue

        roof_data = {}

        # Set structure name and chunk details
        if structure_key == 'all_structures':
            structure_name = "All Structures"
            chunk_id = "C002"
            section_name = "Roof Measurements - All Structures"
        elif structure_key == 'all':
            structure_name = "All Structures"
            chunk_id = "C002"
            section_name = "Roof Measurements"
        else:
            structure_num = structure_key.replace('structure_', '')
            structure_name = f"Structure {structure_num}"
            chunk_id = f"C002_S{structure_num}"
            section_name = f"Roof Measurements - {structure_name}"

        roof_data["structure"] = structure_name

        if "total_area_all_pitches" in measurements:
            roof_data["total_area"] = measurements["total_area_all_pitches"]
        elif "total_roof_area" in summary_measurements and structure_key in ['all', 'all_structures']:
            roof_data["total_area"] = summary_measurements["total_roof_area"]

        if "total_roof_facets" in summary_measurements and structure_key in ['all', 'all_structures']:
            roof_data["total_roof_facets"] = int(summary_measurements["total_roof_facets"])

        if "predominant_pitch" in measurements:
            roof_data["predominant_pitch"] = measurements["predominant_pitch"]
        elif "predominant_pitch" in summary_measurements and structure_key in ['all', 'all_structures']:
            roof_data["predominant_pitch"] = summary_measurements["predominant_pitch"]

        # Add detailed measurements
        measurement_fields = [
            "ridges", "hips", "valleys", "rakes", "eaves_starter", "drip_edge",
            "flashing", "step_flashing", "parapet_walls", "roof_obstructions_perimeter",
            "roof_obstructions_area", "net_roof_area"
        ]

        for field in measurement_fields:
            if field in measurements:
                roof_data[field] = measurements[field]
            elif field in summary_measurements and structure_key in ['all', 'all_structures']:
                roof_data[field] = summary_measurements[field]

        # Calculate net roof area if not present
        if "net_roof_area" not in roof_data and "total_area_all_pitches" in measurements:
            try:
                total_area_str = measurements["total_area_all_pitches"].replace(" sq ft", "").replace(",", "")
                obstruction_area_str = measurements.get("roof_obstructions_area", "0").replace(" sq ft", "").replace(",", "")
                net_area = float(total_area_str) - float(obstruction_area_str)
                roof_data["net_roof_area"] = f"{net_area:.1f} sq ft"
            except (ValueError, AttributeError):
                pass

        if roof_data:
            _add(chunk_id, property_id, section_name, "text", roof_data)
            structure_counter += 1

    # Create pitch breakdown chunks for each structure
    for structure_key, pitch_data in pitch_breakdown.items():
        if not pitch_data:  # Skip empty pitch data
            continue

        if structure_key == 'all_structures':
            # For single structures, use simple names
            has_multiple_structures = any(key.startswith('structure_') for key in pitch_breakdown.keys())
            if has_multiple_structures:
                chunk_id = "C005"
                section_name = "Pitch Breakdown - All Structures"
            else:
                chunk_id = "C005"
                section_name = "Pitch Breakdown"
        elif structure_key == 'all':
            chunk_id = "C005"
            section_name = "Pitch Breakdown"
        else:
            structure_num = structure_key.replace('structure_', '')
            chunk_id = f"C005_S{structure_num}"
            section_name = f"Pitch Breakdown - Structure {structure_num}"

        pitch_chunk_data = {
            "structure": "All Structures" if structure_key in ['all', 'all_structures'] else f"Structure {structure_key.replace('structure_', '')}",
            "pitch_breakdown": pitch_data
        }

        _add(chunk_id, property_id, section_name, "text", pitch_chunk_data)

    # Create waste calculation chunks for each structure
    for structure_key, waste_data in waste_calculation.items():
        if not waste_data:  # Skip empty waste data
            continue

        if structure_key == 'all_structures':
            # For single structures, use simple names
            has_multiple_structures = any(key.startswith('structure_') for key in waste_calculation.keys())
            if has_multiple_structures:
                chunk_id = "C006"
                section_name = "Waste Calculation - All Structures"
            else:
                chunk_id = "C006"
                section_name = "Waste Calculation"
        elif structure_key == 'all':
            chunk_id = "C006"
            section_name = "Waste Calculation"
        else:
            structure_num = structure_key.replace('structure_', '')
            chunk_id = f"C006_S{structure_num}"
            section_name = f"Waste Calculation - Structure {structure_num}"

        waste_chunk_data = {
            "structure": "All Structures" if structure_key in ['all', 'all_structures'] else f"Structure {structure_key.replace('structure_', '')}",
            "waste_calculation": waste_data
        }

        _add(chunk_id, property_id, section_name, "text", waste_chunk_data)

    # Update database with extracted information
    try:
        # Extract report_id for database update
        report_id = Path(pdf_path).stem
        if '_Premium' in report_id:
            report_id = report_id.replace('_Premium', '')
        report_id_match = re.search(r'report_(\d+)', report_id)
        if report_id_match:
            report_id = report_id_match.group(1)

            # Prepare update data
            update_data = {}

            # Add address if found
            if header_data.get('property_address'):
                update_data['address'] = header_data['property_address']

            # Add coordinates if found
            if coordinates.get('latitude'):
                update_data['lat'] = float(coordinates['latitude'])
            if coordinates.get('longitude'):
                update_data['lng'] = float(coordinates['longitude'])

            # Add roof facets if found
            if summary_measurements.get('total_roof_facets'):
                try:
                    update_data['no_of_facets'] = int(summary_measurements['total_roof_facets'])
                except (ValueError, TypeError):
                    pass

            # Add predominant pitch if found (prefer from detailed measurements, fallback to summary)
            predominant_pitch = None
            if detailed_measurements:
                # Check if we have any structure measurements
                for structure_key, measurements in detailed_measurements.items():
                    if measurements.get('predominant_pitch'):
                        predominant_pitch = measurements['predominant_pitch']
                        break

            if not predominant_pitch and summary_measurements.get('predominant_pitch'):
                predominant_pitch = summary_measurements['predominant_pitch']

            if predominant_pitch:
                update_data['predominant_pitch'] = predominant_pitch

            # Update database if we have data to update
            if update_data:
                # Filter out 'address' since update_processing_details doesn't accept it
                # Address can only be set during insert_record
                update_params = {k: v for k, v in update_data.items() if k != 'address'}
                if update_params:
                    # Get the database record ID by report_id first
                    existing_records = PropertyRAGStatusDAO.get_records_by_report_id(report_id)
                    if existing_records and len(existing_records) > 0:
                        record_id = existing_records[0]['id']  # Get the actual database record ID
                        success = PropertyRAGStatusDAO.update_processing_details(
                            record_id=record_id,
                            **update_params
                        )
                        if success:
                            print(f"Updated database record for report {report_id} with extracted data: {list(update_data.keys())}")
                        else:
                            print(f"Failed to update database record for report {report_id}")
                    else:
                        print(f"No database record found for report {report_id} - skipping database update")
                else:
                    print(f"No updatable fields for report {report_id}")

    except Exception as db_error:
        print(f"Error updating database for report extraction: {db_error}")
        # Don't fail the extraction if database update fails

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
    final_chunks_file = final_chunks_dir / f"report_{report_id}.json"

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
