import os
import shutil

def reorganize_dataset(base_dir):

    for class_folder in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_folder)

        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue

        images_folder = os.path.join(class_path, 'images')
        try:
            # Move files from images folder to class folder
            for image_file in os.listdir(images_folder):
                src_path = os.path.join(images_folder, image_file)
                dst_path = os.path.join(class_path, image_file)
                shutil.move(src_path, dst_path)

            # Remove the now empty 'images' folder
            os.rmdir(images_folder)
        except FileNotFoundError:
            continue

def delet_not_img(base_dir):
    for class_folder in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_folder)

        for image_file in os.listdir(class_path):
            src_path = os.path.join(class_path, image_file)
            if src_path.endswith('txt'):
                os.remove(src_path)

reorganize_dataset('../data/tiny-imagenet-200/train')
# Example usage
delet_not_img('../data/tiny-imagenet-200/train')

