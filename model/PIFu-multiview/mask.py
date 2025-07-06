import sys
from rembg import remove
from PIL import Image
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

input_path = args.input
output_path = args.output


input_image = Image.open(input_path)
output_image = remove(input_image)


output_np = np.array(output_image)
if output_np.shape[2] == 4:
    mask = output_np[:, :, 3]  
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(output_path, mask)
else:
    print("Không tìm thấy alpha channel!")