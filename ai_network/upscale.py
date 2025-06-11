import os
import argparse
import torch
import warnings
from PIL import Image
from torchvision import transforms
from UNet import UNet  # Upewnij się, że plik UNet.py jest w tym samym folderze lub popraw import

def load_checkpoints(checkpoint_dir, extension):
    """Searches and sorts model files in the given directory by epoch number."""
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(extension)]
    
    # Sort files - assuming filenames contain epoch numbers
    def extract_number(filename):
        # Extract digits from the filename and convert to integer
        digits = ''.join(filter(str.isdigit, filename))
        return int(digits) if digits else 0  # Default to 0 if no digits are found
    
    files = sorted(files, key=extract_number)
    return [os.path.join(checkpoint_dir, f) for f in files]

def pad_to_multiple(image_tensor, multiple=16):
    """
    Adds padding to image tensor so dimensions are divisible by 'multiple'.
    
    Args:
        image_tensor: Tensor of shape [1, C, H, W]
        multiple: Number by which dimensions should be divisible (default 16)
    
    Returns:
        padded_tensor: Tensor with padding
        original_size: Tuple (H, W) with original dimensions
    """
    _, _, height, width = image_tensor.shape
    original_size = (height, width)
    
    # Calculate required padding
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    
    # Padding is symmetric - split between top/bottom and left/right
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    # PyTorch padding format: (pad_left, pad_right, pad_top, pad_bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    
    # Use reflection padding for better quality at edges
    padded_tensor = torch.nn.functional.pad(image_tensor, padding, mode='reflect')
    
    print(f"Original size: {original_size}, Padded to: {padded_tensor.shape[2:4]}, Padding: {padding}")
    
    return padded_tensor, original_size

def crop_to_original(image_tensor, original_size):
    """
    Crops tensor to original dimensions.
    
    Args:
        image_tensor: Tensor of shape [1, C, H, W]
        original_size: Tuple (H, W) with original dimensions
    
    Returns:
        cropped_tensor: Tensor cropped to original dimensions
    """
    _, _, padded_height, padded_width = image_tensor.shape
    target_height, target_width = original_size
    
    # Calculate offset for centering
    start_h = (padded_height - target_height) // 2
    start_w = (padded_width - target_width) // 2
    
    # Crop tensor
    cropped_tensor = image_tensor[:, :, start_h:start_h + target_height, start_w:start_w + target_width]
    
    print(f"Cropped from {(padded_height, padded_width)} to {cropped_tensor.shape[2:4]}")
    
    return cropped_tensor

def validate_image_size(width, height, min_size=32, max_size=4096):
    """
    Validates if image size is appropriate for processing.
    
    Args:
        width, height: Image dimensions
        min_size: Minimum size (default 32px)
        max_size: Maximum size (default 4096px)
    
    Returns:
        bool: True if size is OK
        str: Error message (if size is not OK)
    """
    if width < min_size or height < min_size:
        return False, f"Image is too small ({width}x{height}). Minimum size: {min_size}x{min_size}"
    
    if width > max_size or height > max_size:
        return False, f"Image is too large ({width}x{height}). Maximum size: {max_size}x{max_size}"
    
    # Check if image won't be too large after padding
    padded_width = width + (16 - width % 16) % 16
    padded_height = height + (16 - height % 16) % 16
    
    if padded_width > max_size or padded_height > max_size:
        return False, f"Image after padding would be too large ({padded_width}x{padded_height}). Use smaller image."
    
    return True, "OK"

def upscale_image(image_path, checkpoint_dir, extension, output_dir):
    """
    Upscales an image using sequential model checkpoints with dynamic resolution support.
    
    Args:
        image_path: Path to input image
        checkpoint_dir: Directory containing model checkpoints
        extension: File extension of model files
        output_dir: Directory to save output images
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load image
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    print(f"Original image size: {original_width}x{original_height}")
    
    # Validate image size
    is_valid, message = validate_image_size(original_width, original_height)
    if not is_valid:
        print(f"ERROR: {message}")
        return
    
    # Input transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]
    
    # Add padding to make dimensions divisible by 16
    padded_input, original_size = pad_to_multiple(input_tensor, multiple=16)

    # Find checkpoints
    checkpoints = load_checkpoints(checkpoint_dir, extension)
    print(f"Found {len(checkpoints)} models.")
    
    if len(checkpoints) == 0:
        print(f"ERROR: No models found in {checkpoint_dir} with extension {extension}")
        return
    
    current_input = padded_input
    os.makedirs(output_dir, exist_ok=True)
    
    # Process sequentially through each model
    for idx, ckpt_path in enumerate(checkpoints, start=1):
        print(f"\nProcessing with model {idx}/{len(checkpoints)}: {os.path.basename(ckpt_path)}")
        
        try:
            # Create model instance - parameters must match those used during training
            model = UNet(dropout=0.3).to(device)
            
            # Load model state
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            with torch.no_grad():
                output_tensor = model(current_input)
            
            # Update input for next model (keep padding)
            current_input = output_tensor
            
            # Crop to original dimensions for saving only
            cropped_output = crop_to_original(output_tensor, original_size)
            
            # Convert tensor to image and save
            output_image = transforms.ToPILImage()(cropped_output.squeeze(0).cpu().clamp(0, 1))
            output_path = os.path.join(output_dir, f"upscaled_after_{idx}.png")
            output_image.save(output_path)
            print(f"Saved result after model {idx}: {output_path}")
            print(f"Saved image size: {output_image.size}")
            
            # Free memory
            del model, checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"ERROR processing model {idx}: {str(e)}")
            continue
    
    print(f"\nProcessing completed. Results saved in folder: {output_dir}")
