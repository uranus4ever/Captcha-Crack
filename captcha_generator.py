from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import string
import imutils
import scipy.misc

CAPTCHA_IMAGE_FOLDER = "./generated_captcha_images/"
OUTPUT_FOLDER = "./extracted_letter_images/"

# Generate random codes, len=4
characters = string.digits + string.ascii_uppercase
print("Using characters: \n" + characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)
generator = ImageCaptcha(width=width, height=height)


def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


def gen_single_letter(batch_size=32):
    new_height, new_width = 20, 20
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    X_letter = np.zeros((batch_size, 4, new_height, new_height, 1), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            letters = extract(X[i], random_str)
            for num in range(4):
                X_letter[i][num] = resize(letters[num], (new_height, new_width))
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X_letter, y


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


def extract(img, str, plot=False):
    '''    
    extract the 4 codes from img
    :param img: PIL Image
    :return: regions contains (x, y, w, h)
    '''
    img_array = np.asarray(img)
    # convert PIL image to cv2 gray image
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    blur_img = blur(thresh)

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(blur_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        if w / h > 1.8:
            # too wide, split it into four letter regions!
            one_fourth_width = int(w / 4)
            letter_image_regions.append((x, y, one_fourth_width, h))
            letter_image_regions.append((x + one_fourth_width, y, one_fourth_width, h))
            letter_image_regions.append((x + one_fourth_width * 2, y, one_fourth_width, h))
            letter_image_regions.append((x + one_fourth_width * 3, y, one_fourth_width, h))
        elif (w / h > 1.3) & (w / h <= 1.8):
            # too wide, split it into three letter regions!
            one_third_width = int(w / 3)
            letter_image_regions.append((x, y, one_third_width, h))
            letter_image_regions.append((x + one_third_width, y, one_third_width, h))
            letter_image_regions.append((x + one_third_width*2, y, one_third_width, h))
        elif (w / h > 0.8) & (w / h <= 1.3):
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    letters = []
    for i in range(len(letter_image_regions)):
        x, y, w, h = letter_image_regions[i]
        image = blur_img[y:y+h, x:x+w]
        letters.append(image)

    if plot:
        # img_new = np.zeros_like(thresh)
        # for (i, (x, y, w, h)) in enumerate(letter_image_regions):
        #     letter_image =
        #     resize_img = resize(letter_image, new_size=(60, 35))

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 2, 1)
        for i in range(len(letter_image_regions)):
            x, y, w, h = letter_image_regions[i]
            cv2.rectangle(img_array, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=1)
        plt.imshow(img_array)
        plt.title(str)
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(thresh, 'gray')
        plt.axis('off')

        for i in range(4):
            plt.subplot(2, 4, i+1+4)
            plt.imshow(resize(letters[i], (30, 20)), cmap='gray')
            plt.axis('off')

        # plt.subplot(2, 2, 3)
        # plt.imshow(blur_img, 'gray')
        # plt.axis('off')
        # plt.subplot(2, 2, 4)
        # plt.imshow()
        # plt.axis('off')

        plt.show()

    return letters


def blur(img, size=5):
    '''
    To filter salt noise with medianBlur
    :param img: PIL image
    :param size: filter size, odd number, eg. 5x5
    :return: 
    '''
    img_array = np.asarray(img)
    return cv2.medianBlur(img_array, size)


def resize(image, new_size):
    return scipy.misc.imresize(image, new_size)


def plotimg(img):
    plt.figure()
    if len(img.shape) > 2:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')


if __name__ == "__main__":
    # random_str, img = generate_code()
    # regions = extract(img, random_str, plot=True)

    X, y = next(gen(1))
    letters = extract(X[0], decode(y), plot=True)


