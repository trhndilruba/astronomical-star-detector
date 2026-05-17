import cv2
import csv


def save_output_image(image):

    cv2.imwrite(
        "../output/detected_stars.png",
        image
    )

    print("Output image saved.")


def save_star_coordinates(contours):

    with open(
        "../output/star_coordinates.csv",
        mode="w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Star_ID",
            "Center_X",
            "Center_Y",
            "Area"
        ])

        for i, contour in enumerate(contours):

            M = cv2.moments(contour)

            if M["m00"] != 0:

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                area = cv2.contourArea(contour)

                writer.writerow([
                    i + 1,
                    cx,
                    cy,
                    area
                ])

    print("Star coordinates saved.")