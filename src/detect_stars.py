import cv2
import numpy as np


def detect_stars(image):

    # adaptive threshold
    thresh = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # küçük noise temizliği
    kernel = np.ones((2,2), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    # contour bul
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # küçük objeleri filtrele
    filtered_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 3:
            filtered_contours.append(contour)

    return filtered_contours, thresh