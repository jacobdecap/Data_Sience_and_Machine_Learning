import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf


#Loads in the data. This is directly in the module
mnist = tf.keras.datasets.mnist

#Retrieve the training data and test data from minst
(x_train,y_train), (x_test,y_test) = mnist.load_data()

# X: pixels on digital canvas
# y: corresponding number associated with input pixel


#Scale X data to be in the range [0,1], this improves model accuracy
#Note: no reason to scale y data as it is the labeled digit
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#Basic Sequential Nerural Network
model = tf.keras.models.Sequential()
#Flatten converts 28x28 grid to 1x784
model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
#Dense Layers are connected to every neuron of previous layer
model.add(tf.keras.layers.Dense(128, activation ='relu'))
model.add(tf.keras.layers.Dense(128, activation ='relu'))
#Will be final output layer with is length 10 for each potential output
model.add(tf.keras.layers.Dense(10, activation ='softmax'))#gives probabilty of each potiental output

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 10)
model.save('scribe.model')

scribe = tf.keras.models.load_model('scribe.model')

loss,accuracy = scribe.evaluate(x_test,y_test)
print(loss)
print(accuracy)

image_number = 1
while os.path.isfile(f"Digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"Digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = scribe.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print('Error! Probably a resolution problem')
    finally:
        image_number += 1


