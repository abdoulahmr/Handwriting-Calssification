import cv2
import numpy as np
from PIL import Image
import os

# Load the image
input_path = "test.tif"
base_output_folder = "test"
os.makedirs(base_output_folder, exist_ok=True)

pil_image = Image.open(input_path).convert('RGB')
image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# --- Row Projection ---
row_sum = np.sum(binary, axis=1)

# Detect rows where there is something
rows = []
in_object = False
start_row = 0
for i, val in enumerate(row_sum):
    if val > 0 and not in_object:
        in_object = True
        start_row = i
    elif val == 0 and in_object:
        in_object = False
        end_row = i
        rows.append((start_row, end_row))

# Now for each row slice, detect columns
for row_idx, (row_start, row_end) in enumerate(rows, start=1):
    line = binary[row_start:row_end, :]
    col_sum = np.sum(line, axis=0)

    # Create folder for this row
    row_folder = os.path.join(base_output_folder, f"row_{row_idx}")
    os.makedirs(row_folder, exist_ok=True)

    in_digit = False
    start_col = 0
    digit_idx = 1
    for j, val in enumerate(col_sum):
        if val > 0 and not in_digit:
            in_digit = True
            start_col = j
        elif val == 0 and in_digit:
            in_digit = False
            end_col = j

            # Crop digit (gray image)
            digit = gray[row_start:row_end, start_col:end_col]

            # Further crop to remove surrounding white
            _, digit_binary = cv2.threshold(digit, 128, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(digit_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                digit_cropped = digit[y:y+h, x:x+w]

                # Skip too small crops (noise)
                if digit_cropped.shape[0] > 5 and digit_cropped.shape[1] > 5:
                    output_path = os.path.join(row_folder, f"digit_{digit_idx}.png")
                    cv2.imwrite(output_path, digit_cropped)
                    digit_idx += 1

    print(f"Saved {digit_idx-1} cleaned digits in '{row_folder}/'")

print(f"All digits cropped tightly and organized by rows inside '{base_output_folder}/'")
