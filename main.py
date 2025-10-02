import cv2
import os
from pathlib import Path
from Interpolation import nearest_neighbor_interpolation, bilinear_interpolation, bicubic_interpolation, lanczos_interpolation, area_based_interpolation, hermite_interpolation
from ai_network.UNet import train
from ai_network.upscale import upscale_image
from argparse import Namespace

#Allowed image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def clean_path(path_string):
    """
    Removes non-printable characters, whitespace from the start and end, and replaces backslashes with forward slashes for compatibility.
    """
    # Remove non-printable characters and whitespace from the start and end
    cleaned = path_string.strip().strip('"').strip("'")
    # Replace backslashes with forward slashes for compatibility
    cleaned = cleaned.replace('\\', '/')
    return cleaned

def is_image_file(file_path):
    """
    Checks if a file is an image based on its extension.
    """
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS

def get_image_files_from_folder(folder_path):
    """
    Returns a list of all image files in the given folder.
    """
    image_files = []
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        return image_files
    
    for file in folder.iterdir():
        if file.is_file() and is_image_file(file):
            image_files.append(str(file))
    
    return sorted(image_files)

def load_image(path):
    """
    Loads an image from given path. If image can't be loaded, prints an error message and returns None.
    """
    try:
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Could not load image: {path}")
        return image
    except Exception as e:
        print(f"Error loading image {path}: {str(e)}")
        return None

def display_image(window_name, image):
    """
    Displays an image in a window with given name.
    Waits for a key press and then destroys the window.
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(path, image):
    """
    Saves an image to given path.
    """
    try:
        # Create a folder if it doesn't exist already
        os.makedirs(os.path.dirname(path), exist_ok=True)
        success = cv2.imwrite(path, image)
        if success:
            print(f"Saved: {path}")
        else:
            print(f"Failed to save: {path}")
        return success
    except Exception as e:
        print(f"Error saving image {path}: {str(e)}")
        return False
    
def get_output_path(input_path, output_dir, suffix="_upscaled"):
    """
    Generates an output path based on the input path.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Create output folder if it doesn't exist already
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add suffix to the filename
    name_without_ext = input_path.stem
    extension = input_path.suffix
    output_filename = f"{name_without_ext}{suffix}{extension}"
    
    return str(output_dir / output_filename)

def process_single_image_with_method(image_path, method_choice, output_dir="./upscaled"):
    """
    Upscales an image using the selected method.
    """
    image = load_image(image_path)
    if image is None:
        return False
    
    output_path = get_output_path(image_path, output_dir)
    
    # Use the selected method
    if method_choice == 1:
        processed = nearest_neighbor_interpolation.nearest_neighbor_interpolation(image)
    elif method_choice == 2:
        processed = bilinear_interpolation.bilinear_interpolation(image)
    elif method_choice == 3:
        processed = bicubic_interpolation.bicubic_interpolation(image)
    elif method_choice == 4:
        processed = lanczos_interpolation.lanczos_interpolation(image)
    elif method_choice == 5:
        processed = area_based_interpolation.area_based_interpolation(image)
    elif method_choice == 6:
        processed = hermite_interpolation.hermite_interpolation(image)
    elif method_choice == 7:
        # U-Net - special case
        model_path = r"./ai_network/checkpoints"
        output_dir_unet = "./upscaled"
        try:
            upscale_image(image_path, model_path, ".pt", output_dir_unet)
            print(f"Image upscaled successfully: {image_path}")
            return True
        except Exception as e:
            print(f"Error upscaling with U-Net: {str(e)}")
            return False
    else:
        print(f"Invalid method choice: {method_choice}")
        return False
    
    # Save the image for methods 1-6
    if method_choice in range(1, 7):
        return save_image(output_path, processed)
    
    return False

def get_interpolation_method():
    """
    Displays a menu and gets the interpolation method choice from the user.
    """
    print("\nSelect interpolation/upscaling method:")
    print("1. Nearest Neighbor Interpolation")
    print("2. Bilinear Interpolation")
    print("3. Bicubic Interpolation")
    print("4. Lanczos Interpolation")
    print("5. Area-based Interpolation")
    print("6. Hermite Interpolation")
    print("7. U-Net Super-Resolution")
    
    while True:
        try:
            choice = int(input("Enter your choice (1-7): "))
            if 1 <= choice <= 7:
                return choice
            else:
                print("Please enter a number between 1 and 7.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def prepare_unet_model():
    """
    Checks if U-Net model exists. If not, trains a new one.
    """
    model_path = r"./ai_network/checkpoints"
    args = Namespace(
        train_low_dir='path/to/low',
        train_high_dir='path/to/high',
        val_low_dir='path/to/low_val',
        val_high_dir='path/to/high_val',
        checkpoint_dir=model_path,
        batch_size=5, epochs=200, lr=0.0005, weight_decay=5e-5,
        dropout=0.3, step_size=10, gamma=0.5,
        log_file='train.log', alpha=0.55, patience=40
    )
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    models_list = [f for f in os.listdir(model_path) if f.endswith(".ph") or f.endswith(".pth")]
    
    if len(models_list) == 0:
        print("No models found in ./ai_network/checkpoints path.")
        print("Training new model...")
        try:
            train(args)
            print("Model trained successfully.")
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    return True


def main():
    """
    Main function of the program. Works as a user interface.
    """
    # Get path and check if it is correct
    raw_path = input("Enter image path or folder path: ")
    path = clean_path(raw_path)
    
    if not path:
        print("Error: Empty path provided.")
        return
    
    # Check if path exists
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"Error: Path does not exist: {path}")
        return
    
    # Check if path is a file or a folder
    image_files = []
    
    if path_obj.is_file():
        if is_image_file(path):
            image_files = [path]
            print(f"Processing single image: {path}")
        else:
            print(f"Error: File is not a supported image format: {path}")
            print(f"Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
            return
    
    elif path_obj.is_dir():
        image_files = get_image_files_from_folder(path)
        if not image_files:
            print(f"No image files found in folder: {path}")
            return
        print(f"Found {len(image_files)} image(s) in folder:")
        for img in image_files:
            print(f"  - {os.path.basename(img)}")
    
    else:
        print(f"Error: Path is neither a file nor a folder: {path}")
        return
    
    # Get interpolation method choice from the user
    choice = get_interpolation_method()
    
    # Prepare U-Net model
    if choice == 7:
        if not prepare_unet_model():
            print("Failed to prepare U-Net model.")
            return
    
    # Ask for confirmation when processing multiple images
    if len(image_files) > 1:
        confirm = input(f"\nProcess all {len(image_files)} images? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Process images
    output_dir = "./upscaled"
    successful = 0
    failed = 0
    
    print(f"\nStarting processing...")
    print("-" * 50)
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_file)}")
        
        success = process_single_image_with_method(image_file, choice, output_dir)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("Processing completed!")
    print(f"Successfully processed: {successful} image(s)")
    if failed > 0:
        print(f"Failed to process: {failed} image(s)")
    print(f"Output directory: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()