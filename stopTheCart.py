import os
import sys
import schp_utils
import extract_clothing
print("Step 1: Generate Segmentation Masks")
os.system("python schp_utils/simple_extractor.py --dataset lip --model-restore checkpoints/final.pth --input-dir input --output-dir output")
print("Step 2: Extract clothing")
os.system("python extract_clothing.py")