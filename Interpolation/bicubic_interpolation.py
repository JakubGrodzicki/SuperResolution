import cv2

def bicubic_interpolation(image):
    """
    Resizes an image using bicubic interpolation.

    Args:
        image (numpy array): The image to be resized.

    Returns:
        numpy array: The resized image.
    """
    return cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
