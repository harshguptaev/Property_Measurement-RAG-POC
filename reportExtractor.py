from reports_client import download_report_file, get_report_info
import os, json, re
from glob import glob


helper_data = {
    "North": 22,
    "South": 23,
    "East": 24,
    "West": 25,
    "Top": 6,
    "pdf":{
        "31": 75,
        "13": 51,
        "17": 69,
        "8": 75,
        "1": 3
    },
    "metadata": 149,
    "DDD": 50
}

def export_data_for_report(report_id: str):
    print(f"Exporting data for report {report_id}")
    # save metadata, all 5 images 
    required_files = ["pdf", "DDD"]
    for file in required_files:
        if file == "pdf":
            pid = get_report_info(report_id)["ProductId"]
            print(f"ProductId: {pid}")
            if str(pid) not in helper_data["pdf"]:
                print(f"ProductId: {pid} not found in helper_data")
                return
            print(f"Downloading report file for ProductId: {pid}")
            download_report_file(report_id, str(helper_data["pdf"][str(pid)]), f"{report_id}.pdf")
            
        else:
            download_report_file(report_id, str(helper_data[file]), f"DDD.png")

    

def extract_measurements_from_pdf(pdf_path: str) -> dict:
    """
    Extract measurement data from the first page of a PDF report.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary containing extracted measurements
    """
    measurements = {}

    try:
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return measurements

        # Extract text from first page
        page = doc[0]
        text = page.get_text()

        if not text:
            doc.close()
            return measurements

        # Define regex patterns for common measurements
        patterns = {
            "total_roof_area": r"Total\s+Roof\s+Area\s*[:=]\s*([\d,]+\.?\d*)\s*(sq\s*ft|sqft|ft²|sq\.?\s*ft\.?)",
            "total_roof_facets": r"Total\s+Roof\s+Facets?\s*[:=]\s*(\d+)",
            "predominant_pitch": r"Predominant\s+Pitch\s*[:=]\s*([0-9]+/[0-9]+|[0-9]+:[0-9]+|[0-9]+\s+in\s+[0-9]+)",
            "no_of_stories": r"(?:No\.?\s+of\s+Stories|Stories|Building\s+Stories)\s*[:=]\s*(\d+)",
            "total_ridges_hips": r"Total\s+Ridges?/Hips?\s*[:=]\s*([\d,]+\.?\d*)\s*(ft|feet|'|\")",
            "total_valleys": r"Total\s+Valleys?\s*[:=]\s*([\d,]+\.?\d*)\s*(ft|feet|'|\")",
            "total_rakes": r"Total\s+Rakes?\s*[:=]\s*([\d,]+\.?\d*)\s*(ft|feet|'|\")",
            "total_eaves": r"Total\s+Eaves?\s*[:=]\s*([\d,]+\.?\d*)\s*(ft|feet|'|\")",
        }

        # Extract measurements using regex
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Clean up the value (remove commas for numbers)
                if key not in ["predominant_pitch"]:
                    try:
                        # Try to convert to float for numeric values
                        value = float(value.replace(',', ''))
                    except ValueError:
                        pass
                measurements[key] = value

        doc.close()

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")

    return measurements


def process_all_pdfs_for_measurements():
    """
    Process all PDF files in ReportsData directory and extract measurements.
    """
    # Find all PDF files in ReportsData and subdirectories
    pdf_files = glob("ReportsData/**/*.pdf", recursive=True)

    for pdf_path in pdf_files:
        try:
            # Extract measurements from PDF
            measurements = extract_measurements_from_pdf(pdf_path)

            if measurements:
                # Create measurementchunks.json in the same directory as the PDF
                pdf_dir = os.path.dirname(pdf_path)
                output_path = os.path.join(pdf_dir, "measurementchunks.json")

                # Save measurements as JSON
                with open(output_path, "w") as f:
                    json.dump(measurements, f, indent=2)

                print(f"Extracted measurements from {pdf_path} to {output_path}")
            else:
                print(f"No measurements found in {pdf_path}")

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")