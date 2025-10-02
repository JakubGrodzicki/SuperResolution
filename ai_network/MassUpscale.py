import os
import argparse
import torch
import warnings
from PIL import Image
from torchvision import transforms
from .UNet import UNet

def load_checkpoints(checkpoint_dir, extension):
    """Searches and sorts model files in the given directory by epoch number."""
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(extension)]
    
    def extract_number(filename):
        digits = ''.join(filter(str.isdigit, filename))
        return int(digits) if digits else 0
    
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

def process_single_image(image_path, checkpoints, output_dir, device):
    """
    Processes a single image through all checkpoints with dynamic resolution support.
    
    Args:
        image_path: Path to input image
        checkpoints: List of checkpoint file paths
        output_dir: Directory to save output images
        device: PyTorch device (cuda/cpu)
    """
    try:
        # Load and validate image
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size
        
        # Validate image size
        is_valid, message = validate_image_size(original_width, original_height)
        if not is_valid:
            print(f"  SKIPPING {os.path.basename(image_path)}: {message}")
            return False
        
        print(f"  Original size: {original_width}x{original_height}")
        
        # Transform to tensor
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Add padding to make dimensions divisible by 16
        padded_input, original_size = pad_to_multiple(input_tensor, multiple=16)
        current_input = padded_input
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Process through all checkpoints
        for idx, ckpt_path in enumerate(checkpoints, start=1):
            try:
                # Load model
                model = UNet(dropout=0.3).to(device)
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                # Process image
                with torch.no_grad():
                    output_tensor = model(current_input)
                
                # Update input for next model (keep padding)
                current_input = output_tensor
                
                # Free model memory
                del model, checkpoint
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"  ERROR with model {idx}: {str(e)}")
                continue
        
        # Crop final output to original dimensions
        cropped_output = crop_to_original(current_input, original_size)
        
        # Save final result
        output_image = transforms.ToPILImage()(cropped_output.squeeze(0).cpu().clamp(0, 1))
        output_path = os.path.join(output_dir, f"{base_name}.png")
        output_image.save(output_path)
        print(f"  Saved result: {output_path} (size: {output_image.size})")
        
        return True
        
    except Exception as e:
        print(f"  ERROR processing {os.path.basename(image_path)}: {str(e)}")
        return False
