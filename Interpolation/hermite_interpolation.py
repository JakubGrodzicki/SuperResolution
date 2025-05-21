import cv2
import numpy as np

def hermite_interpolation(image):
    """
    Resize image using Hermite interpolation, doubling the size (fx=2, fy=2)
    
    Args:
        image: Input OpenCV image (numpy array)
        
    Returns:
        Resized image using Hermite interpolation
    """
    # Get original dimensions
    h, w = image.shape[:2]
    
    # New dimensions (doubled)
    new_h, new_w = h * 2, w * 2
    
    # Create output image
    if len(image.shape) == 3:
        # For color images
        channels = image.shape[2]
        resized = np.zeros((new_h, new_w, channels), dtype=image.dtype)
    else:
        # For grayscale images
        resized = np.zeros((new_h, new_w), dtype=image.dtype)
    
    # Hermite interpolation function
    def hermite(t):
        # Hermite basis functions
        return (2*t*t*t - 3*t*t + 1, 
                t*t*t - 2*t*t + t, 
                -2*t*t*t + 3*t*t, 
                t*t*t - t*t)
    
    # For each pixel in the output image
    for y in range(new_h):
        for x in range(new_w):
            # Map to original image coordinates
            src_x = x / 2
            src_y = y / 2
            
            # Get the four surrounding pixels
            x0 = int(np.floor(src_x))
            y0 = int(np.floor(src_y))
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)
            
            # Calculate fractional component
            tx = src_x - x0
            ty = src_y - y0
            
            # Get hermite coefficients
            hx = hermite(tx)
            hy = hermite(ty)
            
            # Safe access to surrounding pixels with boundary checks
            def safe_access(img, y, x):
                y = max(0, min(y, img.shape[0]-1))
                x = max(0, min(x, img.shape[1]-1))
                return img[y, x]
            
            # Need surrounding points for derivatives
            x_m1 = max(0, x0 - 1)
            y_m1 = max(0, y0 - 1)
            x_p2 = min(w - 1, x1 + 1)
            y_p2 = min(h - 1, y1 + 1)
            
            # Get pixel values
            p00 = safe_access(image, y0, x0)
            p01 = safe_access(image, y0, x1)
            p10 = safe_access(image, y1, x0)
            p11 = safe_access(image, y1, x1)
            
            # Compute x derivatives (using central difference)
            dx_p00 = (safe_access(image, y0, x1) - safe_access(image, y0, x_m1)) / 2.0
            dx_p01 = (safe_access(image, y0, x_p2) - safe_access(image, y0, x0)) / 2.0
            dx_p10 = (safe_access(image, y1, x1) - safe_access(image, y1, x_m1)) / 2.0
            dx_p11 = (safe_access(image, y1, x_p2) - safe_access(image, y1, x0)) / 2.0
            
            # Compute y derivatives (using central difference)
            dy_p00 = (safe_access(image, y1, x0) - safe_access(image, y_m1, x0)) / 2.0
            dy_p01 = (safe_access(image, y1, x1) - safe_access(image, y_m1, x1)) / 2.0
            dy_p10 = (safe_access(image, y_p2, x0) - safe_access(image, y0, x0)) / 2.0
            dy_p11 = (safe_access(image, y_p2, x1) - safe_access(image, y0, x1)) / 2.0
            
            # Compute cross derivatives (using central difference in both directions)
            dxy_p00 = (safe_access(image, y1, x1) - safe_access(image, y1, x_m1) - 
                       safe_access(image, y_m1, x1) + safe_access(image, y_m1, x_m1)) / 4.0
            dxy_p01 = (safe_access(image, y1, x_p2) - safe_access(image, y1, x0) - 
                       safe_access(image, y_m1, x_p2) + safe_access(image, y_m1, x0)) / 4.0
            dxy_p10 = (safe_access(image, y_p2, x1) - safe_access(image, y_p2, x_m1) - 
                       safe_access(image, y0, x1) + safe_access(image, y0, x_m1)) / 4.0
            dxy_p11 = (safe_access(image, y_p2, x_p2) - safe_access(image, y_p2, x0) - 
                       safe_access(image, y0, x_p2) + safe_access(image, y0, x0)) / 4.0
            
            # Setup the coefficient matrix
            if len(image.shape) == 3:
                # For color images
                coeffs = np.zeros((4, 4, channels), dtype=np.float32)
                for c in range(channels):
                    coeffs[0, 0, c] = p00[..., c]
                    coeffs[0, 1, c] = dx_p00[..., c]
                    coeffs[1, 0, c] = dy_p00[..., c]
                    coeffs[1, 1, c] = dxy_p00[..., c]
                    
                    coeffs[0, 2, c] = p01[..., c]
                    coeffs[0, 3, c] = dx_p01[..., c]
                    coeffs[1, 2, c] = dy_p01[..., c]
                    coeffs[1, 3, c] = dxy_p01[..., c]
                    
                    coeffs[2, 0, c] = p10[..., c]
                    coeffs[2, 1, c] = dx_p10[..., c]
                    coeffs[3, 0, c] = dy_p10[..., c]
                    coeffs[3, 1, c] = dxy_p10[..., c]
                    
                    coeffs[2, 2, c] = p11[..., c]
                    coeffs[2, 3, c] = dx_p11[..., c]
                    coeffs[3, 2, c] = dy_p11[..., c]
                    coeffs[3, 3, c] = dxy_p11[..., c]
            else:
                # For grayscale images
                coeffs = np.zeros((4, 4), dtype=np.float32)
                coeffs[0, 0] = p00
                coeffs[0, 1] = dx_p00
                coeffs[1, 0] = dy_p00
                coeffs[1, 1] = dxy_p00
                
                coeffs[0, 2] = p01
                coeffs[0, 3] = dx_p01
                coeffs[1, 2] = dy_p01
                coeffs[1, 3] = dxy_p01
                
                coeffs[2, 0] = p10
                coeffs[2, 1] = dx_p10
                coeffs[3, 0] = dy_p10
                coeffs[3, 1] = dxy_p10
                
                coeffs[2, 2] = p11
                coeffs[2, 3] = dx_p11
                coeffs[3, 2] = dy_p11
                coeffs[3, 3] = dxy_p11
            
            # Compute bicubic interpolation
            result = 0
            for i in range(4):
                for j in range(4):
                    if len(image.shape) == 3:
                        result += np.outer(hy[i], hx[j])[:, :, np.newaxis] * coeffs[i, j]
                    else:
                        result += hy[i] * hx[j] * coeffs[i, j]
            
            # Set the pixel value
            resized[y, x] = np.clip(result, 0, 255).astype(image.dtype)
    
    return resized
