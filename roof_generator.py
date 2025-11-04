#!/usr/bin/env python3
"""
Simplified Roof Generator
Generates roof outline diagrams with length labels and color-coded line types
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import argparse
from typing import List, Dict, Tuple

class RoofGenerator:
    def __init__(self, json_file: str, image_file: str, output_dir: str = "output"):
        """Initialize the roof generator"""
        self.json_file = json_file
        self.image_file = image_file
        self.output_dir = output_dir
        self.predictions = []
        self.image_width = 0
        self.image_height = 0
        self.gsd_factor = 0.07400822647359319  # meters per pixel
        self.feet_conversion = 3.28084  # meters to feet
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        """Load detection results and image data"""
        # Load JSON
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        
        self.predictions = data.get('predictions', [])
        self.image_width = data.get('image_width', 300)
        self.image_height = data.get('image_height', 300)
        
        print(f"Loaded {len(self.predictions)} detections")
        print(f"Image dimensions: {self.image_width}x{self.image_height}")
    
    def calculate_line_length_feet(self, line: List[int]) -> float:
        """Calculate line length in feet"""
        x1, y1, x2, y2 = line
        pixel_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        meters = pixel_length * self.gsd_factor
        feet = meters * self.feet_conversion
        return feet
    
    def create_overlay_with_yellow_borders(self, min_confidence: float = 0.8, include_labels: bool = True):
        """
        Create overlay image with bright yellow borders and optionally length labels
        
        Args:
            min_confidence: Minimum confidence threshold
            include_labels: If True, add length labels. If False, only draw lines.
        """
        filtered_predictions = [p for p in self.predictions if p['score'] >= min_confidence]
        
        # Load original image
        original_img = Image.open(self.image_file)
        img_array = np.array(original_img)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.imshow(img_array)
        
        # Draw lines with bright yellow color
        for i, prediction in enumerate(filtered_predictions):
            line_coords = prediction['line']
            class_name = prediction['class']
            
            x1, y1, x2, y2 = line_coords
            
            # Set line width based on class (increased thickness)
            if class_name.lower() == 'ridge':
                linewidth = 6
            elif class_name.lower() == 'hip':
                linewidth = 5
            elif class_name.lower() == 'eave':
                linewidth = 4
            else:
                linewidth = 4
            
            # Draw bright yellow line
            ax.plot([x1, x2], [y1, y2], 
                   color='yellow',
                   linewidth=linewidth,
                   linestyle='-',
                   alpha=0.9)
            
            # Add length label only if include_labels is True
            if include_labels:
                # Calculate and add length label
                length_feet = self.calculate_line_length_feet(line_coords)
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Offset label position
                offset_x = 10 if i % 2 == 0 else -10
                offset_y = 10 if i % 3 == 0 else -10
                
                ax.text(mid_x + offset_x, mid_y + offset_y, f'{length_feet:.1f}ft', 
                       fontsize=10, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               edgecolor='yellow', alpha=0.9),
                       color='black', fontweight='bold')
        
        # Customize plot (no title)
        ax.set_xlim(0, self.image_width)
        ax.set_ylim(self.image_height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save with appropriate filename based on include_labels
        if include_labels:
            overlay_path = os.path.join(self.output_dir, "roof_overlay_with_lengths.png")
            print(f"Overlay with bright yellow borders and labels saved to: {overlay_path}")
        else:
            overlay_path = os.path.join(self.output_dir, "roof_overlay_without_lengths.png")
            print(f"Overlay with bright yellow borders (no labels) saved to: {overlay_path}")
        
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def create_simplified_outline(self, min_confidence: float = 0.9):
        """Create clean architectural outline"""
        filtered_predictions = [p for p in self.predictions if p['score'] >= min_confidence]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Draw black lines
        for prediction in filtered_predictions:
            line_coords = prediction['line']
            class_name = prediction['class']
            
            # Normalize coordinates
            x1, y1, x2, y2 = (line_coords[0] / self.image_width, 
                             line_coords[1] / self.image_height,
                             line_coords[2] / self.image_width, 
                             line_coords[3] / self.image_height)
            
            # Set line width
            if class_name.lower() == 'ridge':
                linewidth = 6
            elif class_name.lower() == 'hip':
                linewidth = 5
            elif class_name.lower() == 'eave':
                linewidth = 4
            else:
                linewidth = 4
            
            ax.plot([x1, x2], [y1, y2], 
                   color='black', linewidth=linewidth, alpha=1.0)
        
        # Clean up (no title)
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Save
        simplified_path = os.path.join(self.output_dir, "roof_outline_simplified.png")
        plt.tight_layout()
        plt.savefig(simplified_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Simplified outline saved to: {simplified_path}")
    
    def create_simplified_with_lengths(self, min_confidence: float = 0.9):
        """Create simplified outline with length labels"""
        filtered_predictions = [p for p in self.predictions if p['score'] >= min_confidence]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Draw lines and add length labels
        for prediction in filtered_predictions:
            line_coords = prediction['line']
            class_name = prediction['class']
            
            # Normalize coordinates
            x1, y1, x2, y2 = (line_coords[0] / self.image_width, 
                             line_coords[1] / self.image_height,
                             line_coords[2] / self.image_width, 
                             line_coords[3] / self.image_height)
            
            # Set line width
            if class_name.lower() == 'ridge':
                linewidth = 6
            elif class_name.lower() == 'hip':
                linewidth = 5
            elif class_name.lower() == 'eave':
                linewidth = 4
            else:
                linewidth = 4
            
            # Draw black line
            ax.plot([x1, x2], [y1, y2], 
                   color='black', linewidth=linewidth, alpha=1.0)
            
            # Add length label
            length_feet = self.calculate_line_length_feet(line_coords)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Offset to avoid overlapping line
            offset_x = 0.02 if class_name == 'eave' else 0.01
            offset_y = 0.02 if class_name == 'hip' else 0.01
            
            ax.text(mid_x + offset_x, mid_y + offset_y, f'{length_feet:.1f}ft', 
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='black', alpha=0.9),
                   color='black', fontweight='bold')
        
        # Clean up (no title)
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Save
        with_lengths_path = os.path.join(self.output_dir, "roof_outline_simplified_with_lengths.png")
        plt.tight_layout()
        plt.savefig(with_lengths_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Simplified outline with lengths saved to: {with_lengths_path}")
    
    def create_combined_outline_with_types(self, min_confidence: float = 0.9):
        """Create combined outline with color-coded line types"""
        filtered_predictions = [p for p in self.predictions if p['score'] >= min_confidence]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Color mapping for line types
        colors = {'hip': 'red', 'eave': 'blue', 'ridge': 'green', 'flashing': 'orange'}
        
        # Draw lines with colors and add length labels
        for prediction in filtered_predictions:
            line_coords = prediction['line']
            class_name = prediction['class']
            
            # Normalize coordinates
            x1, y1, x2, y2 = (line_coords[0] / self.image_width, 
                             line_coords[1] / self.image_height,
                             line_coords[2] / self.image_width, 
                             line_coords[3] / self.image_height)
            
            # Set line color and width based on type (increased thickness)
            color = colors.get(class_name.lower(), 'black')
            if class_name.lower() == 'ridge':
                linewidth = 6
            elif class_name.lower() == 'hip':
                linewidth = 5
            elif class_name.lower() == 'eave':
                linewidth = 4
            else:
                linewidth = 4
            
            # Draw the colored line
            ax.plot([x1, x2], [y1, y2],
                   color=color, linewidth=linewidth, alpha=1.0)
        
        # Add legend
        legend_elements = []
        for class_name, color in colors.items():
            legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2, label=class_name.title()))
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Clean up (no title)
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Save
        combined_path = os.path.join(self.output_dir, "roof_outline_combined.png")
        plt.tight_layout()
        plt.savefig(combined_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Combined outline with types saved to: {combined_path}")
    
    def generate_all(self):
        """Generate all required diagrams"""
        print("🏠 Roof Generator")
        print("=" * 30)
        
        # Load data
        print("\n1️⃣ Loading data...")
        self.load_data()
        
        # Create overlay with bright yellow borders (with labels)
        print("\n2️⃣ Creating overlay with bright yellow borders and labels...")
        self.create_overlay_with_yellow_borders(0.8, include_labels=True)
        
        # Create overlay without labels
        print("\n2b️⃣ Creating overlay without labels...")
        self.create_overlay_with_yellow_borders(0.8, include_labels=False)
        
        # Create simplified outline
        print("\n3️⃣ Creating simplified architectural outline...")
        self.create_simplified_outline(0.9)
        
        # Create simplified outline with lengths
        print("\n4️⃣ Creating simplified outline with length labels...")
        self.create_simplified_with_lengths(0.9)
        
        # Create combined outline with types
        print("\n5️⃣ Creating combined outline with line types...")
        self.create_combined_outline_with_types(0.9)
        
        print("\n✅ All diagrams created successfully!")
        print(f"\n📁 Output files in '{self.output_dir}' directory:")
        print("   🖼️  roof_overlay_with_lengths.png - Original image with bright yellow roof outline")
        print("   📐 roof_outline_simplified.png - Clean architectural outline")
        print("   📏 roof_outline_simplified_with_lengths.png - Clean outline with length labels")
        print("   🎨 roof_outline_combined.png - Combined outline with color-coded line types")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Roof Generator')
    parser.add_argument('--json', default='../input_files/result.json', help='JSON file with detection results')
    parser.add_argument('--image', default='../input_files/N-67043971.png', help='Original roof image')
    parser.add_argument('--output', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.json):
        print(f"❌ Error: {args.json} not found")
        return
    
    if not os.path.exists(args.image):
        print(f"❌ Error: {args.image} not found")
        return
    
    # Create generator and run
    generator = RoofGenerator(args.json, args.image, args.output)
    generator.generate_all()

if __name__ == "__main__":
    main()

