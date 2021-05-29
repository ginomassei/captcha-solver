import cv2 as cv
import os
import requests
from random import randint


def clean_image(image):
    # Making the image gray.
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Making a binary image.
    ret, thresholded_image = cv.threshold(gray_image, 210, 255, cv.THRESH_BINARY)
    # Median Blur, difuses the straight lines
    blurred_image = cv.medianBlur(thresholded_image, ksize=3)

    return blurred_image


def load_images(path):
    files = os.listdir(path)
    images = []
    
    for file in files:
        element = [cv.imread(str(path + file)), str(file)]
        images.append(element)

    return images


def process_images():
    rawImages = load_images('../data/raw/')
    processedImages = []

    for element in rawImages:
        cleanedImage = clean_image(element[0])
        processedImages.append([cleanedImage, element[1]])

    for element in processedImages:
        cv.imwrite(str('../data/processed-captchas/' + element[1]), element[0])


def get_captcha():
    url = "https://prenotaonline.esteri.it/captcha/default.aspx"
    headers = {
        'User-Agent': 'PostmanRuntime/7.28.0'
    }
    response = requests.get(url, headers=headers)
    
    with open("captcha.jpeg", "wb") as file:
        file.write(response.content)


def crop_digits(element):
    image = element[0]
    filename = element[1]

    # Get the size of the image.
    dimentions = image.shape
    height = dimentions[0]
    width = dimentions[1]

    # The width of a digit is.
    digit_width = 19

    i = 0
    for x in range(22, width, digit_width):
        if x + digit_width > width:
            break
        digit = image[0: height, x: x + digit_width]

        cv.imwrite(f'../data/digits/{filename[i]}-{randint(0, 100000)}.jpeg', digit)
        i += 1


def cropper():
    images = load_images('../data/processed-captchas/')
    
    for image in images:
        crop_digits(image)
