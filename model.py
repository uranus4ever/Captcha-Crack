from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from captcha_generator import *
from imutils import paths
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

width, height, n_len, n_class = 170, 80, 4, len(characters)
new_height, new_width = 20, 20

CAPTCHA_IMAGE_FOLDER = "./my_captcha_img/"
LETTER_IMAGES_FOLDER = "./my_single_letter/"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"


# initialize the data and labels
image_files = paths.list_images(LETTER_IMAGES_FOLDER)
num = len(list(image_files))
# data = np.zeros((num, new_height, new_width, 1), dtype="float")
data = []
labels = []

print("Loading Data...")
# loop over the input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur the image to eliminate noise
    blur_img = blur(image)

    # Resize the letter so it fits in a 20x20 pixel box
    image = resize(blur_img, (new_height, new_width))

    # Add a third channel dimension to the image to make Keras happy
    image = np.expand_dims(image, axis=2)

    # Grab the name of the letter based on the folder it was in
    label = image_file.split(os.path.sep)[-2].split('/')[-1]

    # Add the letter image and it's label to our training data
    # data[i] = image
    data.append(image)
    labels.append(label)


# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)
X_train, Y_train = shuffle(X_train, Y_train)

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(n_class, activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the neural network
result = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=15, verbose=2)

# Save the trained model to disk
model.save(MODEL_FILENAME)
print('model saved as: ', MODEL_FILENAME)

# ### plot the training and validation loss for each epoch
plt.figure(figsize=(10, 5))
plt.plot(result.epoch, result.history['acc'], '-o')
plt.plot(result.epoch, result.history['val_acc'], '-*')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='lower right')
plt.ylim([0, 1])
plt.show()



