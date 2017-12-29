from keras.models import load_model
from captcha_generator import *
from imutils import paths
import numpy as np
import cv2
import pickle


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "./my_captcha_img/"
PREDICT_OUTPUT = "./img/output/"


# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(40,), replace=False)

# loop over the image paths
for image_file in captcha_image_files:
    label = image_file.split('/')[-1]
    label = os.path.splitext(label)[0]

    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    letter_image_regions = extract(image)

    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    if len(letter_image_regions) != 4:
        continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([gray] * 3)
    predictions = []

    # loop over the letters
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[y - 1:y + h + 1, x - 1:x + w + 1]

        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = resize(letter_image, (20, 20))

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0].split('/')[-1]
        predictions.append(letter)

        # draw the prediction on the output image
        cv2.rectangle(image, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(image, letter, (x + 3, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Print the captcha's text
    captcha_text = "".join(predictions)
    print("Predict text is: {}. Label is {}. -- {}".format(captcha_text, label, captcha_text == label))

    # # Show the annotated image
    # cv2.imshow("Output", output)
    # cv2.waitKey()

    # Save the output
    str = ''.join(i for i in predictions)
    _cv2imwrite = cv2.imwrite(PREDICT_OUTPUT+label+".jpg", image)
