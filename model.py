from keras.models import *
from keras.models import load_model
from keras.layers import *
from tqdm import tqdm
# from keras.utils.visualize_util import plot
from IPython.display import Image
from captcha_generator import *


width, height, n_len, n_class = 170, 80, 4, len(characters)
new_height, new_width = 20, 20




# model structure
input_tensor = Input((new_height, new_width, 1))  # binary image



# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(new_height, new_width, 1), activation="relu"))
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
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)

# Save the trained model to disk
model.save(MODEL_FILENAME)



'''
# model structure
input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(4):
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
model = Model(input=input_tensor, output=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# # Visualize the model structure and save picture
# plot(model, to_file="model.png", show_shapes=True)
# Image('model.png')

# model.fit_generator(gen(), samples_per_epoch=5120, nb_epoch=3,
#                     validation_data=gen(), nb_val_samples=128)
#
# model.save('cnn.h5')
# print('Model Saved')

# load weights into new model
model = load_model("cnn.h5")
print("Loaded model from disk")

X, y = next(gen(1))
y_pred = model.predict(X)
plt.figure()
plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
plt.imshow(X[0], cmap='gray')
plt.axis('off')


def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen(1)
    for i in tqdm(range(batch_num)):
        X, y = next(generator)
        y_pred = model.predict(X)
        batch_acc += np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))
    return batch_acc / batch_num

# print('Prediction Accuracy = ', evaluate(model))
'''


