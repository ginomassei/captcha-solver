import cv2 as cv


def clean_image(image):
    # Making the image gray.
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Making a binary image.
    ret, thresholded_image = cv.threshold(gray_image, 210, 255, cv.THRESH_BINARY)
    # Median Blur, difuses the straight lines
    blurred_image = cv.medianBlur(thresholded_image, ksize=3)

    return blurred_image


def crop_digits(image):
    digits = []
    # Get the size of the image.
    dimentions = image.shape
    height = dimentions[0]
    width = dimentions[1]

    # The width of a digit is.
    digit_width = 19

    for x in range(22, width, digit_width):
        if x + digit_width > width:
            break
        digit = image[0: height, x: x + digit_width]
        digits.append(digit)
    return digits
