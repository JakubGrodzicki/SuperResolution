import cv2

def nearest_neighbor_interpolation(image):
    """
    Nearest Neighbor Interpolation
    """
    return cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)