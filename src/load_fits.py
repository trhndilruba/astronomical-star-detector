from astropy.io import fits
import matplotlib.pyplot as plt
import cv2

from preprocess import normalize_image
from preprocess import blur_image

from detect_stars import detect_stars

from utils import save_output_image
from utils import save_star_coordinates


# FITS dosyasını aç
hdul = fits.open("../data/HorseHead.fits")

image_data = hdul[0].data

hdul.close()


# preprocessing
normalized = normalize_image(image_data)

blurred = blur_image(normalized)


# detection
contours, thresh = detect_stars(blurred)


# renkli çıktı oluştur
output = cv2.cvtColor(
    blurred,
    cv2.COLOR_GRAY2BGR
)


# yıldız merkezlerini çiz
for contour in contours:

    M = cv2.moments(contour)

    if M["m00"] != 0:

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(
            output,
            (cx, cy),
            3,
            (0,255,0),
            -1
        )


# çıktı kaydet
save_output_image(output)

# csv kaydet
save_star_coordinates(contours)


# yıldız sayısı
print(f"Detected Stars: {len(contours)}")


# görselleştir
plt.figure(figsize=(10,10))

plt.imshow(output)

plt.title("Detected Stars")

plt.show()