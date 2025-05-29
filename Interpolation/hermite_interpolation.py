import cv2
import numpy as np
from scipy.interpolate import CubicHermiteSpline

def hermite_interpolation(image):
    """
    Resize image using Hermite interpolation with scipy.interpolate.CubicHermiteSpline,
    doubling the size (fx=2, fy=2).

    Args:
        image: Input OpenCV image (numpy array)

    Returns:
        Resized image using Hermite interpolation
    """
    h, w = image.shape[:2]
    new_h, new_w = h * 2, w * 2

    if len(image.shape) == 3:
        channels = image.shape[2]
        resized = np.zeros((new_h, new_w, channels), dtype=image.dtype)
    else:
        resized = np.zeros((new_h, new_w), dtype=image.dtype)

    # Function to compute approximate derivatives for a 1D array
    def compute_derivatives(arr):
        grad = np.zeros_like(arr, dtype=np.float32)
        if len(arr) > 1:
            grad[1:-1] = (arr[2:] - arr[:-2]) / 2.0
            grad[0] = (arr[1] - arr[0]) # Forward difference for first element
            grad[-1] = (arr[-1] - arr[-2]) # Backward difference for last element
        return grad

    num_channels = image.shape[2] if len(image.shape) == 3 else 1
    for c in range(num_channels):
        current_channel = image[:, :, c] if len(image.shape) == 3 else image

        # First, interpolate rows
        interpolated_rows = np.zeros((h, new_w), dtype=np.float32)
        for r in range(h):
            x = np.arange(w)
            y = current_channel[r, :]
            dydx = compute_derivatives(y) # Derivatives along x

            # Create spline
            spl = CubicHermiteSpline(x, y, dydx)

            # Evaluate spline at new x coordinates (0, 0.5, 1, 1.5, ..., w-0.5, w-1)
            new_x = np.linspace(0, w - 1, new_w)
            interpolated_rows[r, :] = spl(new_x)

        # Then, interpolate columns
        interpolated_full = np.zeros((new_h, new_w), dtype=np.float32)
        for col in range(new_w):
            x = np.arange(h)
            y = interpolated_rows[:, col]
            dydx = compute_derivatives(y) # Derivatives along y

            # Create spline
            spl = CubicHermiteSpline(x, y, dydx)

            # Evaluate spline at new y coordinates
            new_y = np.linspace(0, h - 1, new_h)
            interpolated_full[:, col] = spl(new_y)

        # Clip values and assign to the correct channel
        if len(image.shape) == 3:
            resized[:, :, c] = np.clip(interpolated_full, 0, 255).astype(image.dtype)
        else:
            resized = np.clip(interpolated_full, 0, 255).astype(image.dtype)

    return resized
