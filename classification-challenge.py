import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import uuid

training_data_generator = ImageDataGenerator(rescale=1.0/255, zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)
training_iterator = training_data_generator.flow_from_directory("D:/GitHub/classification-challenge/Covid19-dataset/train", class_mode="categorical", color_mode="grayscale", target_size=(256,256), batch_size=2)
test_iterator = test_datagen.flow_from_directory("D:/GitHub/classification-challenge/Covid19-dataset/test", class_mode="categorical", color_mode="grayscale", target_size=(256,256), batch_size=4)

lungs_model = Sequential()
lungs_model.add(keras.Input(shape=(256,256,1)))
lungs_model.add(keras.layers.Conv2D(64, 7, activation='relu'))
lungs_model.add(keras.layers.Conv2D(32, 3, activation='relu'))
lungs_model.add(layers.Flatten())
lungs_model.add(keras.layers.Dense(3, activation='softmax'))
lungs_model.summary()

lungs_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

stop = EarlyStopping(monitor='loss', mode='min', patience=30)
history = lungs_model.fit(training_iterator, validation_data=test_iterator, epochs=50,callbacks=[stop], verbose=1)

lungs_model.evaluate(test_iterator, verbose = 0)

print(history.history.keys())
fig = plt.figure()
plot1 = plt.subplot(2,1,1)
plot1.plot(history.history['loss'])
plot1.plot(history.history['val_loss'])
plot1.set_title('model loss')
plot1.set_ylabel('loss')
plot1.set_xlabel('epoch')
plot1.legend(['train', 'validation'], loc='upper left')


plot2 = plt.subplot(2,1,2)
plot2.plot(history.history['accuracy'])
plot2.plot(history.history['val_accuracy'])
plot2.set_title('model accuracy')
plot2.set_ylabel('accuracy')
plot2.set_xlabel('epoch')
plot2.legend(['train', 'validation'], loc='upper left')

fig.savefig('D:/GitHub/classification-challenge/plots/plot_{}.png'.format(str(uuid.uuid4())))