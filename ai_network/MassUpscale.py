import os
import torch
from PIL import Image
from torchvision import transforms
from .UNet import UNet

def load_checkpoints(checkpoint_dir, extension):
    """
    Loads all model checkpoints from a directory.

    Parameters:
        checkpoint_dir (str): Directory containing model checkpoints.
        extension (str): Extension of the checkpoint files.

    Returns:
        list: List of paths to the loaded checkpoint files, sorted in ascending order by epoch number.
    """
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(extension)]

    def extract_number(filename):
        digits = ''.join(filter(str.isdigit, filename))
        return int(digits) if digits else 0

    files = sorted(files, key=extract_number)
    return [os.path.join(checkpoint_dir, f) for f in files]

def process_single_image(image_path, checkpoints, output_dir, device):
    """
    Processes a single image using multiple U-Net models.

    Parameters:
        image_path (str): Path to the image file to be processed.
        checkpoints (list): List of paths to the model checkpoint files.
        output_dir (str): Directory to save the output images.
        device (torch.device): Device to use for processing (either CPU or GPU).
    """
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    current_input = input_tensor
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for idx, ckpt_path in enumerate(checkpoints, start=1):
        model = UNet(dropout=0.3).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        with torch.no_grad():
            output_tensor = model(current_input)

        current_input = output_tensor
        output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu().clamp(0, 1))
        output_path = os.path.join(output_dir, f"{base_name}_upscaled_after_{idx}.png")
        output_image.save(output_path)
        print(f"Saved: {output_path}")

def upscale_folder(input_dir, checkpoint_dir, extension, output_dir):
    """
    Upscales all images in a given directory using multiple U-Net models.

    Parameters:
        input_dir (str): Directory containing the images to be upscaled.
        checkpoint_dir (str): Directory containing the model checkpoint files.
        extension (str): File extension of the checkpoint files.
        output_dir (str): Directory to save the upscaled images.

    This function processes each image in the input directory by applying a series
    of U-Net models, using the checkpoints found in the specified directory. The 
    results are saved to the output directory.
    """

    checkpoints = load_checkpoints(checkpoint_dir, extension)
    print(f"Found {len(checkpoints)} checkpoint(s).")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in image_extensions:
            print(f"Processing: {filename}...")
            process_single_image(file_path, checkpoints, output_dir, device)
