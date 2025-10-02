# Image Super-Resolution Tool

A comprehensive implementation of various image super-resolution techniques, including traditional interpolation methods and deep learning-based approaches using U-Net architecture. This repository contains the practical implementation of research conducted for a master's thesis on image super-resolution.

## 🎯 Overview

This application enables users to upscale images using multiple approaches:

**Traditional Interpolation Methods:**

- Nearest Neighbor Interpolation
- Bilinear Interpolation
- Bicubic Interpolation
- Lanczos Interpolation
- Area-based Interpolation
- Hermite Interpolation (custom implementation; may run slowly due to computational complexity)

**Deep Learning Approach:**

- **U-Net-based Super-Resolution** with advanced features

## ✨ Key Features

- **Graphical File Selection**: Native OS file/folder selection dialogs (Windows Explorer, macOS Finder, Linux file managers)
- **Interactive CLI**: Simple command-line interface with automatic fallback when GUI is unavailable
- **Automatic File/Folder Detection**: Process single images or entire folders automatically
- **Batch Processing**: Built-in support for processing multiple images at once
- **Intelligent Input Handling**: Automatic cleaning of paths (quotes, spaces, etc.)
- **Command-Line Arguments**: Support for automation and scripting
- **Automatic Model Management**: Detects existing U-Net checkpoints or starts training automatically
- **Sequential Processing**: Uses multiple U-Net checkpoints for progressive enhancement
- **Dynamic Resolution Support**: Handles images of various sizes with intelligent padding
- **Advanced Training Pipeline**: Includes data augmentation, combined loss functions, and early stopping
- **Memory Optimization**: Efficient GPU/CPU memory management
- **Apple Silicon Support**: Optimized training and inference for Apple Silicon processors (M1-M4)
- **CPU Optimization**: Enhanced CPU training performance for systems without dedicated GPUs

## 🛠️ Technical Specifications

### U-Net Architecture

- **Encoder-Decoder Structure** with skip connections
- **Dropout Regularization** (configurable, default: 0.3)
- **Batch Normalization** for stable training
- **Initial Features**: 128 (configurable)
- **Input/Output Channels**: 3 (RGB images)

### Advanced Training Features

- **Joint Data Augmentation**: Synchronized transformations for low/high-res pairs
- **Combined Loss Function**: L1 Loss + Gradient-based Loss for enhanced sharpness
- **Learning Rate Scheduling**: StepLR with configurable decay
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Checkpoint Management**: Automatic saving of best models and periodic checkpoints
- **Multi-Device Support**: Automatic detection and optimization for CUDA, Apple Silicon (MPS), and CPU

### Processing Capabilities

- **Dynamic Resolution**: Supports images from 32×32 to 4096×4096 pixels (Threshold values can be adjusted)
- **Intelligent Padding**: Automatic padding to multiples of 16 for U-Net compatibility
- **Sequential Enhancement**: Progressive upscaling through multiple model checkpoints
- **Memory Safety**: Validation and cleanup to prevent GPU memory overflow
- **Format Support**: JPG, JPEG, PNG, BMP, TIFF, TIF, WEBP

## 📋 Requirements

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible GPU recommended for training (optional for inference)
- **Apple Silicon**: M1-M4 processors supported with MPS acceleration
- **CPU**: Optimized for CPU-only systems
- **GUI Support**: Tkinter (usually included with Python, see platform-specific notes below)

### Core Dependencies

```
torch>=1.12.0
torchvision>=0.13.0
opencv-python>=4.5.0
scipy>=1.7.0
Pillow>=8.0.0
tqdm>=4.60.0
numpy>=1.21.0
```

### Platform-Specific Installation

**Windows:**

```bash
# Tkinter is included with Python by default
pip install -r requirements.txt
```

**macOS:**

```bash
# Tkinter is included with Python by default
pip install -r requirements.txt
```

**Linux (Ubuntu/Debian):**

```bash
# Install tkinter if not present
sudo apt-get update
sudo apt-get install python3-tk

# Install Python dependencies
pip install -r requirements.txt
```

**Linux (Fedora/RHEL/CentOS):**

```bash
# Install tkinter if not present
sudo dnf install python3-tkinter

# Install Python dependencies
pip install -r requirements.txt
```

**Linux (Arch/Manjaro):**

```bash
# Install tkinter if not present
sudo pacman -S tk

# Install Python dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
Image-Super-Resolution-Tool/
├── main.py                     # Main application entry point
├── README.md                   # This file
├── requirements.txt            # Python package dependencies
├── Interpolation/              # Traditional interpolation methods
│   ├── __init__.py
│   ├── nearest_neighbor_interpolation.py
│   ├── bilinear_interpolation.py
│   ├── bicubic_interpolation.py
│   ├── lanczos_interpolation.py
│   ├── area_based_interpolation.py
│   └── hermite_interpolation.py
├── ai_network/                 # Deep learning components
│   ├── __init__.py
│   ├── UNet.py                 # U-Net model and training logic
│   ├── upscale.py             # Image upscaling implementation
│   └── checkpoints/           # Model checkpoints directory
└── upscaled/                  # Output directory for results
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Image-Super-Resolution-Tool
```

### 2. Install Dependencies

```bash
# Check if tkinter is available (Linux users)
python3 -c "import tkinter"

# If error on Linux, install tkinter first (see Platform-Specific Installation above)

# Install Python packages
pip install -r requirements.txt
```

### 3. Run the Application

**Interactive Mode with GUI (Default):**

```bash
python main.py
```

**Interactive Mode without GUI:**

```bash
python main.py --no-gui
```

**Command-Line Mode (Automation):**

```bash
# Process single image
python main.py --input image.jpg --method 3 --output ./results

# Process entire folder
python main.py --input ./images --method 7 --output ./upscaled
```

### 4. Follow the Interactive Prompts

When running in interactive mode:

1. **GUI Mode (default)**:
   - A dialog appears asking to select "Folder", "File", or "Cancel"
   - Choose your input type and browse to select
2. **CLI Mode (fallback or --no-gui)**:

   - Enter the path manually when prompted
   - Supports paths with quotes and spaces

3. Select upscaling method (1-7)
4. For folders, confirm batch processing
5. Wait for processing to complete
6. Find results in the `./upscaled/` directory

## 💡 Usage Examples

### GUI Mode (Default)

```bash
$ python main.py

==================================================
            IMAGE SUPER-RESOLUTION TOOL
==================================================
Opening file selection dialog...
[GUI dialog appears - user selects "File" then chooses image.jpg]
Processing single image: /home/user/image.jpg

Select interpolation/upscaling method:
1. Nearest Neighbor Interpolation
2. Bilinear Interpolation
3. Bicubic Interpolation
4. Lanczos Interpolation
5. Area-based Interpolation
6. Hermite Interpolation
7. U-Net Super-Resolution
Enter your choice (1-7): 3

Processing completed!
Output directory: /path/to/upscaled
```

### Command-Line Arguments (Automation)

```bash
# Process with specific method
python main.py --input photo.jpg --method 2 --output ./enhanced

# Batch process folder
python main.py --input ./vacation_photos --method 7 --output ./enhanced_photos

# Force CLI mode even with GUI available
python main.py --no-gui
```

### Batch Processing (Folder)

```bash
$ python main.py
[User selects folder via GUI]

Found 15 image(s) in folder:
  - IMG_001.jpg
  - IMG_002.jpg
  - ...

Select interpolation/upscaling method:
1. Nearest Neighbor Interpolation
...
7. U-Net Super-Resolution
Enter your choice (1-7): 7

Process all 15 images? (y/n): y

Starting processing...
--------------------------------------------------
[1/15] Processing: IMG_001.jpg
Saved: ./upscaled/IMG_001_upscaled.jpg
[2/15] Processing: IMG_002.jpg
...

==================================================
Processing completed!
Successfully processed: 15 image(s)
Output directory: /path/to/upscaled
```

### Path Input Examples (CLI Mode)

The program automatically handles various path formats:

```bash
# With quotes (automatically removed)
"C:\Users\John\Pictures\photo.jpg"

# Without quotes
/home/user/images/photo.png

# Relative paths
./images/
../photos/vacation.jpg

# With spaces
"C:\My Documents\My Pictures\image.jpg"
```

## 🖥️ Command-Line Arguments

| Argument   | Short | Description                | Example              |
| ---------- | ----- | -------------------------- | -------------------- |
| `--input`  | `-i`  | Input image or folder path | `--input ./images`   |
| `--method` | `-m`  | Upscaling method (1-7)     | `--method 3`         |
| `--output` | `-o`  | Output directory           | `--output ./results` |
| `--no-gui` |       | Disable GUI file selection | `--no-gui`           |

## 📊 Dataset Configuration

For U-Net training, you need paired low-resolution and high-resolution images.

### Recommended Dataset

**Kaggle Image Super Resolution Dataset**: https://www.kaggle.com/datasets/adityachandrasekhar/image-super-resolution

### Configuration Setup

Edit the dataset paths in `main.py`:

```python
args = Namespace(
    train_low_dir='path/to/your/low_resolution_training_images',
    train_high_dir='path/to/your/high_resolution_training_images',
    val_low_dir='path/to/your/low_resolution_validation_images',
    val_high_dir='path/to/your/high_resolution_validation_images',
    # ... other parameters
)
```

### Dataset Structure

```
your_dataset/
├── train/
│   ├── low/        # Low-resolution training images
│   └── high/       # High-resolution training images
└── validation/
    ├── low/        # Low-resolution validation images
    └── high/       # High-resolution validation images
```

## ⚙️ Configuration Parameters

### Training Parameters

| Parameter      | Default | Description             |
| -------------- | ------- | ----------------------- |
| `batch_size`   | 5       | Batch size for training |
| `epochs`       | 200     | Maximum training epochs |
| `lr`           | 0.0005  | Learning rate           |
| `weight_decay` | 5e-5    | L2 regularization       |
| `dropout`      | 0.3     | Dropout probability     |
| `step_size`    | 10      | LR scheduler step size  |
| `gamma`        | 0.5     | LR decay factor         |
| `alpha`        | 0.55    | Gradient loss weight    |
| `patience`     | 40      | Early stopping patience |

### Model Architecture Parameters

- **Initial Features**: 128
- **Encoder Levels**: 4 (with max pooling)
- **Decoder Levels**: 4 (with transpose convolution)
- **Skip Connections**: Between corresponding encoder-decoder levels

## 🔧 Advanced Usage

### Programmatic Batch Processing

You can extend the functionality for custom batch processing:

```python
from pathlib import Path
import cv2

def process_images_with_filter(input_dir, output_dir, method=3):
    """Process all images in a directory with a specific method"""
    input_path = Path(input_dir)

    for image_file in input_path.glob("*.jpg"):
        # Your custom processing logic here
        image = cv2.imread(str(image_file))
        # Apply selected method
        processed = apply_method(image, method)
        # Save with custom naming
        output_path = Path(output_dir) / f"{image_file.stem}_enhanced.jpg"
        cv2.imwrite(str(output_path), processed)
```

### Custom Training Configuration

Modify training parameters directly in `main.py`:

```python
args = Namespace(
    # Dataset paths
    train_low_dir='your/train/low/path',
    train_high_dir='your/train/high/path',
    val_low_dir='your/val/low/path',
    val_high_dir='your/val/high/path',

    # Training parameters
    batch_size=8,           # Increase if you have more GPU memory
    epochs=300,             # Adjust based on your needs
    lr=0.001,              # Experiment with different learning rates
    dropout=0.2,           # Lower dropout for larger datasets
    alpha=0.3,             # Adjust gradient loss contribution
    patience=50            # Patience for early stopping
)
```

## 📈 Model Training Details

### Loss Function

The training uses a **Combined Loss Function**:

- **L1 Loss**: For pixel-wise accuracy
- **Gradient Loss**: For edge preservation and sharpness
- **Weighting**: Configurable balance via `alpha` parameter

### Data Augmentation

Applied jointly to low/high-resolution pairs:

- **Horizontal Flipping** (20% probability)
- **Rotation** (-5° to +5°)
- **Color Jittering**: Brightness, contrast, saturation, hue

### Training Process

1. **Automatic Detection**: Checks for existing checkpoints
2. **Training Initialization**: If no models found, starts training automatically
3. **Progress Monitoring**: Real-time loss tracking and logging
4. **Checkpoint Saving**: Best model and periodic saves
5. **Early Stopping**: Prevents overfitting
6. **Memory Management**: Automatic GPU memory cleanup

## 📂 Output Structure

### Single Image Processing

```
upscaled/
└── image_name_upscaled.png    # Processed image with suffix
```

### Batch Processing

```
upscaled/
├── image1_upscaled.png
├── image2_upscaled.png
├── image3_upscaled.png
└── ...
```

### Sequential U-Net Processing

```
upscaled/
├── image_upscaled_after_1.png    # Result after first model
├── image_upscaled_after_2.png    # Result after second model (if available)
└── ...                           # Progressive results
```

## 🐛 Troubleshooting

### Common Issues

**1. Tkinter Not Available**

```
Error: ImportError: No module named 'tkinter'
Solution:
- Linux: sudo apt-get install python3-tk (Ubuntu/Debian)
- macOS/Windows: Should be included with Python
- Alternative: Use --no-gui flag to bypass GUI
```

**2. CUDA Out of Memory**

```
Solution: Reduce batch_size in args or use CPU mode
```

**3. No Models Found**

```
Solution: Ensure dataset paths are correct; training will start automatically
```

**4. Image Too Large / Image Too Small 📏**

Currently, the **maximum supported size** is $4096 \times 4096$ pixels, and the **minimum size** is $32 \times 32$ pixels.

---

**ℹ️ Customizing Image Size Limits**

If you need to change these limits, you can edit the `validate_image_size` function in the **`upscale.py`** file.

The parameters to modify are:

- `min_size` (default $32$)
- `max_size` (default $4096$)

The program automatically adjusts the size so that it is divisible by 16.

**5. GUI Dialog Not Appearing**

```
Possible causes:
- Running through SSH without X forwarding
- Using WSL without X server
- Running in Docker container
Solution: Use --no-gui flag or set up X forwarding
```

**6. Path Not Found**

```
Solution: Check if path exists and remove any extra quotes or spaces
The program now handles quotes automatically
```

### Performance Optimization

**GPU Training:**

- Use CUDA-compatible GPU for faster training
- Adjust batch size based on available GPU memory
- Monitor GPU utilization during training

**Apple Silicon (M1-M4):**

- Automatic MPS acceleration detection and usage
- Optimized memory management for unified memory architecture
- Recommended for Mac users with Apple Silicon

**CPU Training:**

- Expect longer training times
- Consider reducing model complexity for CPU-only setups
- Use smaller batch sizes
- Optimized performance for multi-core processors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Based on master's thesis research in image super-resolution
- U-Net architecture inspired by medical image segmentation applications
- Dataset recommendations from Kaggle community
- Traditional interpolation methods implemented using OpenCV and SciPy

## 📧 Support

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

**Note**: This tool is designed for research and educational purposes. For production use, consider additional optimizations and testing with your specific use cases.
