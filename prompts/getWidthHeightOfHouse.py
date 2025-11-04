# PROMPT = r"""
# You are an expert in aerial imagery and computer vision.
# You are given a top-down aerial image that may contain one or more buildings.
# The image width is {{image_width}} pixels and height is {{image_height}} pixels.


# Your task:
# Find the main roofed building that is fully visible in the image.
# Do NOT include any building that is partially cut off and make sure that the central house covering major part of the image is present in the image.


# Return only a JSON object with two fields:
# {
# "width_px": <int>,
# "height_px": <int>
# }
# These should represent the tight bounding box width and height in pixels
# of the entire visible main house.


# Do not include any explanations, coordinates, or text — only the JSON.
# """


# PROMPT = r"""
# You are an expert in aerial and ortho imagery analysis. 
# Given a top-down ortho image of a property, identify the rectangular region 
# that tightly encloses the entire building roof, including a 50-pixel buffer on each side. 
# Based on that region, suggest the most efficient cropped image size (width × height in pixels) 
# that fully contains the buffered roof and minimizes unnecessary background.
# As a User Prompt Please use:-
# Here is an {{image_width}}×{{image_height}} pixel ortho image of a building.
# Detect the main building footprint, apply a 50 px buffer around it, and suggest the minimal crop dimension that fully contains it.
# Return only a JSON object with the suggested crop dimensions and bounding box coordinates, for example:
# {
#   "width_px": <int>,
#   "height_px": <int>
# }
# """

PROMPT = r"""
You are an expert in aerial and ortho imagery analysis and geometric reasoning. 
You are analyzing a top-down ortho image of a property. 
Your goal is to identify the rectangular bounding region that fully encloses the *entire building roof*, including all extensions, shadows, or roof overhangs — ensuring that no part of the roof or attached structures is cropped out.

After identifying the tight bounding box of the roof, expand it by an additional 50 pixels on all sides as a buffer to include the roof edges and gutters. 
If the building is rotated or diagonal, the bounding box should still fully contain the rotated roof projection.
Use the buffered region to determine the most efficient cropped image size (width × height in pixels) that covers this area completely while minimizing background.

Always make sure the coordinates are clipped within the image boundaries.

Mandaory Condition:

- Please make sure entire building is covered in the crop. 
- If any part of the building is not covered, 
Example:
  - if right side or left side is not covered, then you need to increase the width of the crop accordingly.
  - If top or bottom is not covered, then you need to increase the height of the crop accordingly.
  - If any cornor is not covered, then you need to increase the width and height of the crop accordingly.
  - If you found any other part of building then reduce the width and height but make sure the reduce in width and height should not affect the original building coverage in the crop.

As a User Prompt Please use:-
Here is an {{image_width}}×{{image_height}} pixel ortho image of a property.

Identify the main building or roof structure, determine a bounding rectangle that covers the entire visible roof, 
then add a 50-pixel buffer in all directions. 
Ensure no roof corners, dormers, or attached sections are cropped out.

Finally, provide the minimal crop size (width_px × height_px) that fits this buffered region, 
keeping the crop within the original {{image_width}}×{{image_height}} frame.


Return only a JSON object with the suggested crop dimensions, for example:
{
  "width_px": <int>,
  "height_px": <int>,
}
"""