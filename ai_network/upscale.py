import os
import torch
from PIL import Image
from torchvision import transforms
from .UNet import UNet

def load_checkpoints(checkpoint_dir, extension):
    """Searches and sorts model files in the given folder by epoch number."""
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(extension)]
    
    # Sorting files – assumes filenames contain the epoch number
    def extract_number(filename):
        # Extract digits from the filename and convert to integer
        digits = ''.join(filter(str.isdigit, filename))
        return int(digits) if digits else 0  # Default to 0 if no digits are found
    
    files = sorted(files, key=extract_number)
    return [os.path.join(checkpoint_dir, f) for f in files]

def upscale_image(image_path, checkpoint_dir, extension, output_dir):
    # Prepare the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Input transformation – assumes models expect tensors in range [0,1]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Load Image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]

    # Find Checkpoints
    checkpoints = load_checkpoints(checkpoint_dir, extension)
    print(f"Found {len(checkpoints)} model(s).")
    
    current_input = input_tensor
    os.makedirs(output_dir, exist_ok=True)
    
    # Process sequentially through each model
    for idx, ckpt_path in enumerate(checkpoints, start=1):
        # Create model instance – parameters must match those used during training
        model = UNet(dropout=0.3).to(device)
        # Load model state
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            output_tensor = model(current_input)
        
        # Update input for the next model
        current_input = output_tensor
        
        # Convert tensor to image and save
        output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu().clamp(0, 1))
        output_path = os.path.join(output_dir, f"upscaled_after_{idx}.png")
        output_image.save(output_path)
        print(f"Saved result from model {idx}: {output_path}")
