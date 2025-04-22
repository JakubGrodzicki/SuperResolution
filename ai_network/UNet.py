import os
import argparse
import logging
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import gc
import random

# UNet Model Definition
class UNet(nn.Module):
    def __init__(self, dropout, in_channels=3, out_channels=3, init_features=128):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1", dropout=dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.encoder2 = UNet._block(features, features*2, name="enc2", dropout=dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.encoder3 = UNet._block(features*2, features*4, name="enc3", dropout=dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.encoder4 = UNet._block(features*4, features*8, name="enc4", dropout=dropout)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.bottleneck = UNet._block(features*8, features*16, name="bottleneck", dropout=dropout)

        self.upconv4 = nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block(features*16, features*8, name="dec4", dropout=dropout)
        self.upconv3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block(features*8, features*4, name="dec3", dropout=dropout)
        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block(features*4, features*2, name="dec2", dropout=dropout)
        self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features*2, features, name="dec1", dropout=dropout)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name, dropout):
        return nn.Sequential(OrderedDict([
            (f"{name}_conv1", nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False
            )),
            (f"{name}_batchnorm1", nn.BatchNorm2d(features)),
            (f"{name}_relu1", nn.ReLU(inplace=True)),
            (f"{name}_dropout", nn.Dropout2d(p=dropout)),
            (f"{name}_conv2", nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False
            )),
            (f"{name}_batchnorm2", nn.BatchNorm2d(features)),
            (f"{name}_relu2", nn.ReLU(inplace=True)),
            (f"{name}_dropout2", nn.Dropout2d(p=dropout))
        ]))

# Dataset Class
class ImagePairDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        
        self.low_images = sorted(os.listdir(low_dir))
        self.high_images = sorted(os.listdir(high_dir))
        
        assert len(self.low_images) == len(self.high_images), "Mismatched dataset sizes"
        for low, high in zip(self.low_images, self.high_images):
            assert low == high, "Mismatched image pairs"

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        try:
            low_img = Image.open(os.path.join(self.low_dir, self.low_images[idx])).convert('RGB')
            high_img = Image.open(os.path.join(self.high_dir, self.high_images[idx])).convert('RGB')
            
            if self.transform is not None:
                low_img, high_img = self.transform(low_img, high_img)
            
            return TF.to_tensor(low_img), TF.to_tensor(high_img)
            
        except FileNotFoundError as e:
            print(f"Skipping {self.low_images[idx]}: {e}")
            return None

def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, checkpoint_dir, filename, logger, extra_message=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    file_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, file_path)
    if extra_message:
        logger.info(extra_message)
    else:
        logger.info(f'Checkpoint saved at epoch {epoch+1} as {filename}.')

# Custom joint transformations class
class JointTransforms:
    def __init__(self):
        self.brightness_range = (0.9, 1.1)  # 1 ± 0.1
        self.contrast_range = (0.9, 1.1)
        self.saturation_range = (0.9, 1.1)
        self.hue_range = (-0.1, 0.1)

    def __call__(self, low, high):
        # Random horizontal flipping
        if random.random() < 0.2:
            low = TF.hflip(low)
            high = TF.hflip(high)

        # Random rotation
        angle = random.uniform(-5, 5)
        low = TF.rotate(low, angle)
        high = TF.rotate(high, angle)

        # Generate color jitter parameters
        brightness_factor = random.uniform(*self.brightness_range)
        contrast_factor = random.uniform(*self.contrast_range)
        saturation_factor = random.uniform(*self.saturation_range)
        hue_factor = random.uniform(*self.hue_range)

        # Apply color jitter
        low = TF.adjust_brightness(low, brightness_factor)
        low = TF.adjust_contrast(low, contrast_factor)
        low = TF.adjust_saturation(low, saturation_factor)
        low = TF.adjust_hue(low, hue_factor)

        high = TF.adjust_brightness(high, brightness_factor)
        high = TF.adjust_contrast(high, contrast_factor)
        high = TF.adjust_saturation(high, saturation_factor)
        high = TF.adjust_hue(high, hue_factor)

        return low, high
    
class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize Sobel kernels for 1 input channel
        self.sobel_x = torch.tensor([[ -1, 0, 1], 
                                     [ -2, 0, 2], 
                                     [ -1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[ -1, -2, -1], 
                                     [  0,  0,  0], 
                                     [  1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, output, target):
        self.sobel_x = self.sobel_x.to(output.device)
        self.sobel_y = self.sobel_y.to(output.device)
        
        # Compute gradients for each channel (R, G, B)
        grad_out_x = torch.nn.functional.conv2d(output, self.sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        grad_out_y = torch.nn.functional.conv2d(output, self.sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
        grad_output = torch.sqrt(grad_out_x**2 + grad_out_y**2 + 1e-6)
        
        grad_tgt_x = torch.nn.functional.conv2d(target, self.sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        grad_tgt_y = torch.nn.functional.conv2d(target, self.sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
        grad_target = torch.sqrt(grad_tgt_x**2 + grad_tgt_y**2 + 1e-6)
        
        # Average gradients across channels
        return torch.mean(torch.abs(grad_output - grad_target))

# Perceptual Loss using VGG16
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        # Extract features up to 'relu2_2' (layers 0-8)
        self.vgg_layers = nn.Sequential(*list(vgg.features.children())[:9])
        # Freeze parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.vgg_layers.eval()
        
        # Normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.criterion = nn.L1Loss()
    
    def forward(self, output, target):
        # Normalize images
        output = (output - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # Extract features
        features_output = self.vgg_layers(output)
        features_target = self.vgg_layers(target)
        
        # Compute loss
        return self.criterion(features_output, features_target)

class CombinedLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.gradient = GradientLoss()
        #self.perceptual = PerceptualLoss()
        self.alpha = alpha # Adjust the ratio of L1 loss and loss (less means more L1 weight) 
        
    def forward(self, output, target):
        return self.l1(output, target) + self.alpha * self.gradient(output, target)
    
# Training Function
def train(args):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    transform = JointTransforms()

    # Create datasets
    logger.info(f"Loading images from {args.train_low_dir} and {args.train_high_dir}")
    train_dataset = ImagePairDataset(args.train_low_dir, args.train_high_dir, transform=transform)
    logger.info(f"Loaded {len(train_dataset)} images for training")
    logger.info(f"Loading images from {args.val_low_dir} and {args.val_high_dir}")
    val_dataset = ImagePairDataset(args.val_low_dir, args.val_high_dir)
    logger.info(f"Loaded {len(val_dataset)} images for validation")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = UNet(dropout=args.dropout).to(device)
    criterion = CombinedLoss(alpha=args.alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler: Decay LR every 'step_size' epochs by gamma factor
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_val_loss = float('inf')  # Track the best validation loss
    early_stop_counter = 0       # Counter for early stopping
    
    scaler = torch.amp.GradScaler()

    try:
        # Training loop
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            
            # Training phase
            for low, high in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
                low = low.to(device)
                high = high.to(device)
                
                optimizer.zero_grad()

                with torch.amp.autocast(device.type, enabled=True):
                    outputs = model(low)
                    loss = criterion(outputs, high)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item() * low.size(0)
            
            # Calculate average training loss
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for low, high in val_loader:
                    low = low.to(device)
                    high = high.to(device)
                    outputs = model(low)
                    val_loss += criterion(outputs, high).item() * low.size(0)
            
            val_loss /= len(val_loader.dataset)
            
            # Log progress
            logger.info(f'Epoch {epoch+1}/{args.epochs} - '
                        f'Train Loss: {train_loss:.8f} - '
                        f'Val Loss: {val_loss:.8f} - '
                        f'Difference: {train_loss - val_loss:.8f}')
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                save_checkpoint(
                    epoch, model, optimizer, train_loss, val_loss,
                    args.checkpoint_dir, 'best_checkpoint.pt', logger,
                    extra_message=f'New best validation loss {val_loss:.8f} at epoch {epoch+1}. Best checkpoint saved as best_checkpoint.pt.'
                )
            else:
                early_stop_counter += 1
                if early_stop_counter >= args.patience:
                    logger.info(f'Early stopping at epoch {epoch+1}.')
                    save_checkpoint(
                        epoch, model, optimizer, train_loss, val_loss,
                        args.checkpoint_dir, f'checkpoint_{epoch+1}.pt', logger,
                        extra_message=f'Last epoch: {epoch+1} saved as checkpoint_{epoch+1}.pt.'
                    )
                    break

            if (epoch + 1) % 30 == 0:
                save_checkpoint(
                    epoch, model, optimizer, train_loss, val_loss,
                    args.checkpoint_dir, f'checkpoint_{epoch+1}.pt', logger
                )
                logger.info(f'Best val_loss: {best_val_loss:.8f}')

            scheduler.step()
        
        logger.info('Training completed.')
    except KeyboardInterrupt:
        logger.info('Training interrupted by user.')
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info('GPU memory cleared.')
        gc.collect()
        logger.info('CPU memory cleared.')