# Roof Report Generator

A comprehensive system for generating professional PDF reports from roof measurement data.

## Features

- **Automated Data Extraction**: Processes JSON files, images, and measurement data
- **Professional PDF Reports**: Generates high-quality reports with charts, images, and detailed measurements
- **Batch Processing**: Handle multiple properties at once
- **Flexible Templates**: Customizable HTML templates for different report styles
- **Image Integration**: Automatically includes aerial views, facet analysis, and directional images

## Quick Start

1. **Setup Environment**:
   ```bash
   cd pdf_generation
   python setup.py
   ```

2. **Generate Single Report**:
   ```bash
   python integrated_report_generator.py 38.422999_-90.403464
   ```

3. **Batch Process Multiple Properties**:
   ```bash
   python integrated_report_generator.py --batch folder1 folder2 folder3
   ```

## Data Folder Structure

Each property data folder should contain:

```
38.422999_-90.403464/
├── final_data.json          # Property info and summary measurements
├── area.json                # Facet area calculations
├── pitch_response.json      # Pitch measurements by direction
├── response.json            # Facet detection results
├── pictometry_response.json # Image metadata and coordinates
├── gserve_response.json     # Geographic data
├── top.jpg                  # Aerial top view
├── response-top.png         # Facet analysis overlay
├── response-top-lengths.png # Measurements overlay
├── *_cropped.png           # Directional view images
└── top/south/north/east/west/ # Detailed directional images
```

## Generated Reports Include

### Property Information
- Address and coordinates
- Report ID and generation date
- Property metadata

### Roof Measurements
- Total roof area
- Predominant pitch
- Number of facets, ridges, eaves, etc.

### Detailed Facet Analysis
- Individual facet areas and percentages
- Confidence scores
- Visual representations

### Pitch Analysis
- Pitch measurements by direction (North, South, East, West)
- Pitch ratios and angles
- Average pitch calculations

### Visual Documentation
- Aerial photography
- Facet detection overlays
- Measurement annotations
- Directional views

## Command Line Options

```bash
python integrated_report_generator.py [OPTIONS] DATA_FOLDER

Options:
  --output, -o DIR          Output directory (default: output)
  --template, -t FILE       Template file (default: ROOF_REPORT_TEMPLATE.html)
  --templates-dir DIR       Templates directory (default: templates)
  --batch, -b FOLDERS       Process multiple folders

Examples:
  # Single property
  python integrated_report_generator.py 38.422999_-90.403464

  # Custom output directory
  python integrated_report_generator.py 38.422999_-90.403464 -o reports

  # Batch processing
  python integrated_report_generator.py --batch prop1 prop2 prop3

  # Custom template
  python integrated_report_generator.py data_folder -t custom_template.html
```

## File Structure

```
pdf_generation/
├── integrated_report_generator.py  # Main report generator
├── data_extractor.py              # Data processing utilities
├── roof_report_generator.py       # Legacy generator (deprecated)
├── requirements.txt               # Python dependencies
├── setup.py                      # Setup script
├── README.md                     # This file
├── templates/
│   ├── ROOF_REPORT_TEMPLATE.html # Main report template
│   ├── EV_ROOFING.html           # Original template
│   └── EV_ROOFING_FRONTPAGE.html # Front page template
└── output/                       # Generated reports
```

## Dependencies

- **weasyprint**: PDF generation from HTML
- **jinja2**: Template rendering
- **Pillow**: Image processing
- **System dependencies**: Cairo, Pango, GDK-Pixbuf (for WeasyPrint)

## Troubleshooting

### WeasyPrint Installation Issues

**macOS**:
```bash
brew install cairo pango gdk-pixbuf libffi
pip install weasyprint
```

**Ubuntu/Debian**:
```bash
sudo apt-get install libcairo2-dev libpango1.0-dev libgdk-pixbuf2.0-dev libffi-dev
pip install weasyprint
```

### Common Issues

1. **Missing data files**: Ensure all required JSON files are present
2. **Image loading errors**: Check image file formats and paths
3. **Template errors**: Verify template syntax and variable names
4. **PDF generation fails**: Check WeasyPrint system dependencies

## Customization

### Creating Custom Templates

1. Copy `ROOF_REPORT_TEMPLATE.html` to a new file
2. Modify HTML structure and CSS styling
3. Use Jinja2 template syntax for dynamic content
4. Available template variables:
   - `property_info`: Address, coordinates, dates
   - `measurements`: Areas, pitches, counts
   - `facets`: Individual facet details
   - `pitch_data`: Directional pitch measurements
   - `images`: Base64-encoded images

### Template Variables

```python
{
    'property_info': {
        'address': str,
        'latitude': float,
        'longitude': float,
        'report_date': str,
        'report_id': str
    },
    'measurements': {
        'total_area': float,
        'predominant_pitch': float,
        'num_facets': int,
        # ... other counts
    },
    'facets': [
        {
            'facet_number': int,
            'area_sqft': float,
            'percentage': float,
            'confidence_score': float
        }
    ],
    'pitch_data': {
        'directions': {
            'north/south/east/west': {
                'pitch_value': float,
                'pitch_ratio': str,
                'pitch_angle': float
            }
        }
    },
    'images': {
        'main_images': {
            'top_view': str,  # base64 data URL
            'response_top': str,
            # ... other images
        }
    }
}
```

## License

This project is part of the Property Measurement RAG POC system.
