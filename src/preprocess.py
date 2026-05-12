import numpy as np
import cv2


def normalize_image(image):

    # minimum değeri çıkar
    image = image - np.min(image)

    # 0-1 aralığına getir
    image = image / np.max(image)

    # 0-255 aralığına çevir
    image = (image * 255).astype(np.uint8)

    return image


def blur_image(image):

    blurred = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    return blurred