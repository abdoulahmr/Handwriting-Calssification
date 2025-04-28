import cv2
import numpy as np
from PIL import Image
import os

# Load TIFF file
input_path = "app.tif"
output_folder = "segmented_digits"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open image with PIL and convert to RGB if needed
pil_image = Image.open(input_path).convert('RGB')

# Convert PIL Image to OpenCV format
image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours left-to-right (optional, useful if numbers are horizontally aligned)
bounding_boxes = [cv2.boundingRect(c) for c in contours]
contours = [c for _, c in sorted(zip(bounding_boxes, contours), key=lambda b: b[0][0])]

# Save each contour as a separate image
for idx, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    
    # Add a little margin
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w += 2 * margin
    h += 2 * margin

    # Crop the digit
    digit = gray[y:y+h, x:x+w]

    # Save the cropped digit
    output_path = os.path.join(output_folder, f"digit_{idx+1}.png")
    cv2.imwrite(output_path, digit)

print(f"Segmented {len(contours)} digits and saved to '{output_folder}/'")
