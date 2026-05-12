import cv2


def detect_stars(image):

    # threshold uygula
    _, thresh = cv2.threshold(
        image,
        180,
        255,
        cv2.THRESH_BINARY
    )

    # contour bul
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours, thresh