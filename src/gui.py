import sys
import cv2

from astropy.io import fits

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QFileDialog
)

from PyQt5.QtCore import Qt

from preprocess import normalize_image
from preprocess import blur_image

from detect_stars import detect_stars


class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.image_path = None

        self.setWindowTitle(
            "Astronomical Star Detector"
        )

        self.resize(900, 700)

        self.setup_ui()

    def setup_ui(self):

        layout = QVBoxLayout()

        # Başlık
        self.title_label = QLabel(
            "Astronomical Star Detection System"
        )

        self.title_label.setAlignment(
            Qt.AlignCenter
        )

        # FITS yükleme butonu
        self.load_button = QPushButton(
            "Load FITS Image"
        )

        self.load_button.clicked.connect(
            self.load_image
        )

        # Detection butonu
        self.detect_button = QPushButton(
            "Detect Stars"
        )

        self.detect_button.clicked.connect(
            self.run_detection
        )

        # Görsel / sonuç alanı
        self.image_label = QLabel(
            "Image Preview"
        )

        self.image_label.setAlignment(
            Qt.AlignCenter
        )

        # Layout'a ekle
        layout.addWidget(self.title_label)

        layout.addWidget(self.load_button)

        layout.addWidget(self.detect_button)

        layout.addWidget(self.image_label)

        self.setLayout(layout)

    def load_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select FITS File",
            "",
            "FITS Files (*.fits)"
        )

        if file_path:

            self.image_path = file_path

            self.image_label.setText(
                f"Loaded File:\n{file_path}"
            )

    def run_detection(self):

        if not self.image_path:

            self.image_label.setText(
                "Please load a FITS image first."
            )

            return

        # FITS dosyasını aç
        hdul = fits.open(self.image_path)

        image_data = hdul[0].data

        hdul.close()

        # preprocessing
        normalized = normalize_image(image_data)

        blurred = blur_image(normalized)

        # detection
        contours, thresh = detect_stars(
            blurred
        )

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
                    (0, 255, 0),
                    -1
                )

        # çıktı kaydet
        cv2.imwrite(
            "../output/gui_result.png",
            output
        )

        # kullanıcıya bilgi ver
        self.image_label.setText(
            f"Detected Stars: {len(contours)}\n\n"
            f"Result saved to:\noutput/gui_result.png"
        )


app = QApplication(sys.argv)

window = MainWindow()

window.show()

sys.exit(app.exec_())