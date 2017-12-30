from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import string
import imutils
import scipy.misc
import os
import glob

CAPTCHA_IMAGE_FOLDER = "./my_captcha_img/"
OUTPUT_FOLDER = "./my_single_letter/"

# Generate random codes, len=4
characters = string.digits + string.ascii_uppercase
print("Using characters: \n" + characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)
generator = ImageCaptcha(width=width, height=height)


def save_captcha(num=100):
    for i in range(num):
        print("Saving images...{}/{}".format(i+1, num))

        random_str = ''.join([random.choice(characters) for j in range(4)])
        img = generator.generate_image(random_str)

        save_path = os.path.join(CAPTCHA_IMAGE_FOLDER, random_str)

        # write the image to a file
        p = os.path.join(CAPTCHA_IMAGE_FOLDER, "{}.png".format(str(random_str)))
        _save = cv2.imwrite(p, np.asarray(img))


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


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


def extract(img, str='CODE', plot=False):
    '''    
    extract the 4 codes from img
    :param img: cv2 imread BGR Image
    :return: regions contains (x, y, w, h)
    '''

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

        if (w < 15) & (h < 20):
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
            cv2.rectangle(img, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=1)
        plt.imshow(img)
        plt.title(str)
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(thresh, 'gray')
        plt.axis('off')

        for i in range(4):
            plt.subplot(2, 4, i+1+4)
            plt.imshow(resize(letters[i], (30, 20)), cmap='gray')
            plt.axis('off')

        plt.show()

    return letter_image_regions


def blur(img, size=7):
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
    save_captcha(num=10)
    # Get a list of all the captcha images we need to process
    captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
    counts = {}

    # loop over the image paths
    for (i, captcha_image_file) in enumerate(captcha_image_files):
        print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

        # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
        # grab the base filename as the text
        filename = os.path.basename(captcha_image_file)
        captcha_correct_text = os.path.splitext(filename)[0]

        # Load the image and convert it to grayscale
        image = cv2.imread(captcha_image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        letter_image_regions = extract(image, str=captcha_correct_text)

        # If we found more or less than 4 letters in the captcha, our letter extraction
        # didn't work correcly. Skip the image instead of saving bad training data!
        if len(letter_image_regions) != 4:
            continue

        # Sort the detected letter images based on the x coordinate to make sure
        # we are processing them from left-to-right so we match the right image
        # with the right letter
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        # Save out each letter as a single image
        for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
            # Grab the coordinates of the letter in the image
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = gray[y - 1:y + h + 1, x - 1:x + w + 1]

            # Get the folder to save the image in
            save_path = os.path.join(OUTPUT_FOLDER, letter_text)

            # if the output directory does not exist, create it
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # write the letter image to a file
            count = counts.get(letter_text, 1)
            p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
            cv2.imwrite(p, letter_image)

            # increment the count for the current key
            counts[letter_text] = count + 1
