import cv2
import os
from Interpolation import nearest_neighbor_interpolation, bilinear_interpolation, bicubic_interpolation, lanczos_interpolation, area_based_interpolation, hermite_interpolation
from ai_network.UNet import train
from ai_network.upscale import upscale_image
#from ai_network.MassUpscale import process_single_image, load_checkpoints
from argparse import Namespace

def load_image(path):
    """
    Loads an image from given path. If image can't be loaded, prints an error message and exits.
    """
    image = cv2.imread(path)
    if image is None:
        print("Could not load image: " + path)
        exit(1)
    return image

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
    cv2.imwrite(path, image)

def main():
    """
    Main function of the program. Works as a user interface.
    """
    path = input("Enter image path: ")
    image = load_image(path)
    save_path = r"./upscaled/upscaled_image.png"
    model_path = r"./ai_network/checkpoints"
    args = Namespace(
        train_low_dir='path/to/low',
        train_high_dir='path/to/high',
        val_low_dir='path/to/low_val',
        val_high_dir='path/to/high_val',
        checkpoint_dir= model_path,
        batch_size=5, epochs=200,lr=0.0005, weight_decay=5e-5,
        dropout=0.3, step_size=10, gamma=0.5,
        log_file='train.log', alpha=0.55, patience=40
    )
    if image is None:
        print("Could not load image: " + path)
        return
    
    print("Select option with witch you want to upscale the image:")
    print("1. Nearest Neighbor Interpolation")
    print("2. Bilinear Interpolation")
    print("3. Bicubic Interpolation")
    print("4. Lanczos Interpolation")
    print("5. Area-based Interpolation")
    print("6. Hermite Interpolation")
    print("7. U-Net Super-Resolution")
    print("8. U-Net Super-Resolution Mass Upscaling - WIP")

    choice = int(input("Enter your choice (1-7): "))
    if choice == 1:
        image = nearest_neighbor_interpolation.nearest_neighbor_interpolation(image)
        save_image(save_path, image)
    elif choice == 2:
        image = bilinear_interpolation.bilinear_interpolation(image)
        save_image(save_path, image)
    elif choice == 3:
        image = bicubic_interpolation.bicubic_interpolation(image)
        save_image(save_path, image)
    elif choice == 4:
        image = lanczos_interpolation.lanczos_interpolation(image)
        save_image(save_path, image)
    elif choice == 5:
        image = area_based_interpolation.area_based_interpolation(image)
        save_image(save_path, image)
    elif choice == 6:
        image = hermite_interpolation.hermite_interpolation(image)
        save_image(save_path, image)
    elif choice == 7:
        try:
            # Find all checkpoint files in ./ai_network/checkpoints path
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            models_list = [f for f in os.listdir(model_path) if f.endswith(".ph") or f.endswith(".pth")]
            if len(models_list) == 0:
                print("No models found in ./ai_network/checkpoints path.")
                print("Training new model...")
                train(args)
                print("Model trained successfully.")
                print("Upscaling image with U-Net model...")
                output_dir = "./upscaled"
                upscale_image(path, model_path, ".pt", output_dir)
                print("Image upscaled successfully.")
            else:
                pass
        except Exception as e:
            print("Error: " + str(e))
    elif choice == 8:
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            # Find all checkpoint files in ./ai_network/checkpoints path
            models_list = [f for f in os.listdir(model_path) if f.endswith(".ph") or f.endswith(".pth")]
            if len(models_list) == 0:
                print("No models found in ./ai_network/checkpoints path.")
            else:
                pass
        except Exception as e:
            print("Error: " + str(e))
        finally:
            print("FEATURE WIP")
    else:
        print("Invalid choice. Exiting...")
        return

if __name__ == "__main__":
    main()