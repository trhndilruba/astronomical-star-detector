from astropy.io import fits
import matplotlib.pyplot as plt
import cv2

from preprocess import normalize_image
from preprocess import blur_image
from detect_stars import detect_stars


# FITS dosyasını aç
hdul = fits.open("../data/HorseHead.fits")

# görüntü datasını al
image_data = hdul[0].data

# dosyayı kapat
hdul.close()


# normalize et
normalized = normalize_image(image_data)

# blur uygula
blurred = blur_image(normalized)


# yıldız tespiti
contours, thresh = detect_stars(blurred)


# renkli görüntü oluştur
output = cv2.cvtColor(
    blurred,
    cv2.COLOR_GRAY2BGR
)


# yıldızları işaretle
for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        1
    )


# yıldız sayısı
print(f"Detected Stars: {len(contours)}")


# sonucu göster
plt.figure(figsize=(10,10))

plt.imshow(output)

plt.title("Detected Stars")

plt.show()