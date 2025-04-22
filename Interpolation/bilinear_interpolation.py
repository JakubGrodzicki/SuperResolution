import cv2

def bilinear_interpolation(image):
    """
    Resizes an image using bilinear interpolation.

    Args:
        image (numpy array): The image to be resized.

    Returns:
        numpy array: The resized image.
    """
    return cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

