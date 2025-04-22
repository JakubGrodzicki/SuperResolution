# Image Super-Resolution Tool

This repository contains code implementing the topics discussed in my master's thesis on image super-resolution.

## Overview

This application allows users to upscale images using various interpolation methods as well as a deep learning-based approach using a U-Net model. Users can provide an image file and choose one of the following upscaling methods:

- Nearest Neighbor Interpolation
- Bilinear Interpolation
- Bicubic Interpolation
- Lanczos Interpolation
- Area-based Interpolation
- Hermite Interpolation
- **U-Net-based Super-Resolution**

## Features

- Simple command-line interface to choose the upscaling method.
- Automatic detection of available U-Net model checkpoints.
- If no model is found in `./ai_network/checkpoints/`, the application will automatically start training a new U-Net model.
- Trained models are reused to upscale new images.

## Requirements

Make sure you have the required dependencies installed. This includes:

- Python 3.8+
- OpenCV
- PyTorch
- (Optional) Access to a GPU for faster model training

## Dataset for Training

To train the U-Net model, you need to provide paths to your dataset of low- and high-resolution image pairs.

It is recommended to use the dataset available here:  
📦 [Image Super Resolution Dataset on Kaggle](https://www.kaggle.com/datasets/adityachandrasekhar/image-super-resolution)

## Configuration

Before using U-Net training, edit the following lines in `main.py` to point to your dataset:

```python
args = Namespace(
    train_low_dir='path/to/low',
    train_high_dir='path/to/high',
    val_low_dir='path/to/low_val',
    val_high_dir='path/to/high_val',
    ...
)
```

## How to Use

1. Run the script:

   ```bash
   python main.py
   ```

2. Enter the path to the image you want to upscale.

3. Choose the desired upscaling method by selecting a number (1–7):

   - If option 7 (U-Net Super-Resolution) is selected:
     - If no trained model exists in `./ai_network/checkpoints/`, training will begin automatically.
     - Once a model is available, it will be used to upscale the image.

## Output

Upscaled images are saved in the `./upscaled/` directory.

## U-Net Architecture

The U-Net used in this project is a deep convolutional neural network with encoder-decoder structure, skip connections, and dropout regularization. It is enhanced by:

- **Joint data augmentation** (flipping, rotation, color jitter)
- **Gradient-based loss function** combined with L1 loss for sharpness
- (Optional) **Perceptual loss** based on VGG16 features

## Batch Processing with AI (Mass Upscaling)
In addition to upscaling single images, the system supports mass upscaling of all images in a folder using multiple U-Net models. This functionality is provided by the `MassUpscale.py` module, which is imported in `main.py` as `upscale_folder`.

To enable mass upscaling directly from `main.py`, you can extend the `main()` function with the following code snippet:

```python
elif choice == 8:
    input_dir = input("Enter folder path with images to upscale: ")
    output_dir = "./upscaled_batch"
    upscale_folder(input_dir, model_path, ".pt", output_dir)
    print("Batch upscaling completed successfully.")
```

Then, update the options prompt to include:
```python
print("8. U-Net Super-Resolution for Folder (Batch Mode)")
```

This addition allows users to choose between single image upscaling and batch processing directly from the same interface.

## License

This project is licensed under the MIT License.

