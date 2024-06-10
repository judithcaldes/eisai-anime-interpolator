import os
import argparse
from PIL import Image
from tqdm import tqdm

def resize_image(image_path, output_path, size=(960, 540)):
    """Resize image to specified size"""
    img = Image.open(image_path)
    img = img.resize(size)
    img.save(output_path)

def convert_to_png(image_path):
    """Convert image to PNG format"""
    img = Image.open(image_path)
    new_path = os.path.splitext(image_path)[0] + '.png'
    img.save(new_path, 'PNG')
    return new_path

def rename_folders_and_images(input_folder, output_folder):
    """Rename folders and images as per the specified format"""
    folder_count = 1
    image_count = 1
    num_folders = len(os.listdir(input_folder))
    with tqdm(total=num_folders, desc='Processing folders') as pbar:
        for folder_name in sorted(os.listdir(input_folder)):
            folder_path = os.path.join(input_folder, folder_name)
            if os.path.isdir(folder_path):
                new_folder_name = f"{image_count:010}-{image_count+1:010}-{image_count+2:010}"
                new_folder_path = os.path.join(output_folder, new_folder_name)
                os.makedirs(new_folder_path, exist_ok=True)
                image_count += 3

                for image_name in sorted(os.listdir(folder_path)):
                    image_path = os.path.join(folder_path, image_name)
                    if os.path.isfile(image_path):
                        # Convert to PNG if not already in PNG format
                        if not image_name.lower().endswith('.png'):
                            image_path = convert_to_png(image_path)
                        resize_image(image_path, os.path.join(new_folder_path, f"{folder_count:010}.png"))
                        folder_count += 1
                pbar.update(1)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('input', type=str, help='Path to input folder')
    ap.add_argument('output', type=str, help='Path to output folder')
    args = ap.parse_args()

    input_folder = args.input
    output_folder = args.output

    rename_folders_and_images(input_folder, output_folder)
