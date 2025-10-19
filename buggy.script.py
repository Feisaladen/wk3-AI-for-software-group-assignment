# mnist_buggy.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2                # using OpenCV directly (may cause dtype issues)
import matplotlib.pyplot as plt

# 1) Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# BUG A: forgetting channel dimension and using float64 implicitly
x_train = x_train / 255          # becomes float64 on some setups
x_test  = x_test / 255

# 2) Build model (bug: wrong input shape, missing Flatten, wrong final activation for logits)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28)),  # should be (28,28,1)
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dense(64, activation='relu'),   # BUG: Dense used without Flatten -> shape mismatch
    layers.Dense(10)                       # BUG: no activation -> logits but we'll use wrong loss
])

# 3) Compile (bug: using categorical_crossentropy with sparse integer labels)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# 4) Bad preprocessing using OpenCV on arrays: mixing shapes and types
def add_rotation_with_cv2(images):
    out = []
    for im in images:
        im8 = (im * 255).astype(np.uint8)    # if im is already uint8 this double-scales
        rot = cv2.getRotationMatrix2D((14,14), 10, 1)
        rimg = cv2.warpAffine(im8, rot, (28,28))
        out.append(rimg)
    return np.array(out)

x_test_rot = add_rotation_with_cv2(x_test)  # sometimes results in shape (N,28,28) dtype=uint8

# 5) Train (bug: using mismatched shapes and wrong label format)
# Also passing x_test_rot (uint8) to fit as validation_data (dtype mismatch)
model.fit(x_train, y_train, epochs=2, validation_data=(x_test_rot, y_test))

# 6) Predict (bug: model.predict returns logits; code treats them as probabilities)
sample = x_test[:5]
pred = model.predict(sample)
pred_labels = np.argmax(pred)  # BUG: np.argmax without axis will collapse incorrectly

print("Predicted:", pred_labels)
print("True:     ", y_test[:5])
