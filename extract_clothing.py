import cv2
import numpy as np
import os

# Define directories
input_dir = 'input'
output_dir = 'output'
extracted_dir = 'extracted'

# Create the extracted directory if it doesn’t exist
if not os.path.exists(extracted_dir):
    os.makedirs(extracted_dir)

# Function to get the color palette for LIP dataset
def get_palette(num_cls):
    palette = [0] * (num_cls * 3)
    for j in range(0, num_cls):
        lab = j
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

# Get palette for LIP dataset (20 classes)
palette = get_palette(20)

# Clothing labels from the LIP dataset
clothing_labels = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19]

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Split filename into base and extension
        base, ext = os.path.splitext(filename)
        
        # Define paths
        original_path = os.path.join(input_dir, filename)
        # Assuming masks are saved as .png with the same base name
        mask_path = os.path.join(output_dir, base + '.png')
        
        # Create unique filenames for each output type
        binary_mask_filename = base + '_binary_mask.png'
        extracted_filename = base + '_extracted' + ext
        
        binary_mask_path = os.path.join(extracted_dir, binary_mask_filename)
        extracted_path = os.path.join(extracted_dir, extracted_filename)

        # Load the original image
        original = cv2.imread(original_path)
        if original is None:
            print(f"Failed to load {original_path}")
            continue

        # Load the mask as a color image
        mask_color = cv2.imread(mask_path)
        if mask_color is None:
            print(f"Failed to load {mask_path}")
            continue

        # Convert mask from BGR to RGB
        mask_rgb = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)

        # Create label map from the colored mask
        label_map = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
        for label in range(20):
            color = (palette[label*3], palette[label*3+1], palette[label*3+2])
            mask_label = np.all(mask_rgb == color, axis=2)
            label_map[mask_label] = label

        # Print unique values in the label map for debugging
        print(f"Label map unique values for {filename}: {np.unique(label_map)}")

        # Create a binary mask for clothing labels (0 or 255)
        binary_mask = np.isin(label_map, clothing_labels).astype(np.uint8) * 255

        # Save the binary mask with a unique name as PNG
        cv2.imwrite(binary_mask_path, binary_mask)
        print(f"Saved binary mask to {binary_mask_path}")

        # Extract clothing using the binary mask
        extracted = cv2.bitwise_and(original, original, mask=binary_mask)

        # Save the extracted clothing image with a unique name
        cv2.imwrite(extracted_path, extracted)
        print(f"Saved extracted image to {extracted_path}")