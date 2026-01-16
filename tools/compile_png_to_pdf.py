#!/usr/bin/env python3
"""
Compile PNG images from a directory into a single PDF.
"""

import argparse
from pathlib import Path
from PIL import Image

def images_to_pdf(image_dir: Path, output_pdf: Path):
    """
    Convert all PNG images in image_dir to a single PDF.
    """
    images = [Image.open(p) for p in sorted(image_dir.glob("*.png"))]
    if not images:
        print("No PNG images found.")
        return

    images[0].save(output_pdf, save_all=True, append_images=images[1:], resolution=100.0)
    print(f"PDF created: {output_pdf}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compile PNG images into a PDF.")
    ap.add_argument("--image_dir", type=str, required=True, help="Directory containing PNG images.")
    ap.add_argument("--output_pdf", type=str, required=True, help="Output PDF file path.")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    output_pdf = Path(args.output_pdf)
    images_to_pdf(image_dir, output_pdf)