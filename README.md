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
- Hermite Interpolation (custom implementation)

**Deep Learning Approach:**

- **U-Net-based Super-Resolution** with advanced features

## ✨ Key Features

- **Interactive CLI**: Simple command-line interface for method selection
- **Automatic Model Management**: Detects existing U-Net checkpoints or starts training automatically
- **Sequential Processing**: Uses multiple U-Net checkpoints for progressive enhancement
- **Dynamic Resolution Support**: Handles images of various sizes with intelligent padding
- **Advanced Training Pipeline**: Includes data augmentation, combined loss functions, and early stopping
- **Batch Processing**: Mass upscaling capabilities for processing entire folders
- **Memory Optimization**: Efficient GPU/CPU memory management

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

### Processing Capabilities

- **Dynamic Resolution**: Supports images from 32×32 to 4096×4096 pixels
- **Intelligent Padding**: Automatic padding to multiples of 16 for U-Net compatibility
- **Sequential Enhancement**: Progressive upscaling through multiple model checkpoints
- **Memory Safety**: Validation and cleanup to prevent GPU memory overflow

## 📋 Requirements

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible GPU recommended for training (optional for inference)

### Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
scipy>=1.7.0
Pillow>=8.0.0
tqdm>=4.60.0
numpy>=1.21.0
```

Install all dependencies:

```bash
pip install torch torchvision opencv-python scipy Pillow tqdm numpy
```

## 📁 Project Structure

```
Image-Super-Resolution-Tool/
├── main.py                     # Main application entry point
├── README.md                   # This file
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
│   ├── upscale.py             # Single image upscaling
│   ├── MassUpscale.py         # Batch processing functionality
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
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python main.py
```

### 4. Follow the Interactive Prompts

1. Enter the path to your image
2. Select upscaling method (1-7)
3. Wait for processing to complete
4. Find results in the `./upscaled/` directory

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

### Batch Processing (Programmatic)

For processing entire folders, you can extend the main function:

```python
from ai_network.MassUpscale import upscale_folder

# Process entire folder
input_directory = "path/to/input/images"
output_directory = "path/to/output/images"
checkpoint_directory = "./ai_network/checkpoints"

upscale_folder(input_directory, checkpoint_directory, ".pt", output_directory)
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

## 📁 Output Structure

### Single Image Processing

```
upscaled/
└── upscaled_after_1.png    # Result after first model
└── upscaled_after_2.png    # Result after second model (if available)
└── ...                     # Progressive results
```

### Batch Processing

```
upscaled_batch/
├── image1.png
├── image2.png
└── ...
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```
Solution: Reduce batch_size in args or use CPU mode
```

**2. No Models Found**

```
Solution: Ensure dataset paths are correct; training will start automatically
```

**3. Image Too Large**

```
Solution: Maximum supported size is 4096×4096 pixels
```

**4. Import Errors**

```
Solution: Ensure all dependencies are installed correctly
```

### Performance Optimization

**GPU Training:**

- Use CUDA-compatible GPU for faster training
- Adjust batch size based on available GPU memory
- Monitor GPU utilization during training

**CPU Training:**

- Expect longer training times
- Consider reducing model complexity for CPU-only setups
- Use smaller batch sizes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

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
