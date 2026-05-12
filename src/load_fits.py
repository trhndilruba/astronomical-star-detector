from astropy.io import fits
import matplotlib.pyplot as plt

from preprocess import normalize_image
from preprocess import blur_image


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

# görüntüyü göster
plt.figure(figsize=(10, 10))

plt.imshow(blurred, cmap='gray')

plt.title("Preprocessed Astronomical Image")

plt.colorbar()

plt.show()