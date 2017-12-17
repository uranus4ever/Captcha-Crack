from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import string
import imutils

# Generate random codes, len=4
characters = string.digits + string.ascii_uppercase
print("Using characters: \n" + characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)
generator = ImageCaptcha(width=width, height=height)


def generate_code(plot=False):
    random_str = ''.join([random.choice(characters) for j in range(4)])
    img = generator.generate_image(random_str)
    if plot:
        plt.imshow(img)
        plt.title(random_str)
        img_save_path = './captcha_img/'
        # plt.savefig(img_save_path + random_str + '.png')
    return random_str, img


def extract(img, plot=False):
    '''    
    extract the 4 codes from img
    :param img: PIL Image
    :return: 
    '''
    # convert PIL image to cv2 gray image
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        if (w < 10) & (h < 10):
            continue

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    if plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        for i in range(len(letter_image_regions)):
            x, y, w, h = letter_image_regions[i]
            cv2.rectangle(np.asarray(img), (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(gray, 'gray')
        plt.axis('off')
        plt.show()

    return
