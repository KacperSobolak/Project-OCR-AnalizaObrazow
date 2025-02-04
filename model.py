from tensorflow.keras import layers, models
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

width = 50
height = 100
channel = 1

def load_data():
    images = np.array([]).reshape(0, height, width)
    labels = np.array([])

    dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10,
                  'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20,
                  'L': 21, 'M': 22, 'N': 23, 'P': 24, 'Q': 25, 'R': 26, 'S': 27, 'T': 28, 'U': 29, 'V': 30,
                  'W': 31, 'X': 32, 'Y': 33, 'Z': 34}

    directories = [directory for directory in glob.glob('LettersImages/*')]

    for directory in directories:  
        all_files_in_subdir = glob.glob(directory + '/*.jpg') 
        images_from_current_subdir = np.array([np.array(Image.open(file)) for file in all_files_in_subdir])
        labels_from_current_subdir = [dictionary[directory[-1]]] * len(images_from_current_subdir) 
        images = np.append(images, images_from_current_subdir, axis=0)
        labels = np.append(labels, labels_from_current_subdir, axis=0)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)
    return (x_train, y_train), (x_test, y_test)


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channel)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(units=35, activation='softmax'))   
    return model


(train_images, train_labels), (test_images, test_labels) = load_data()
train_images = train_images.reshape((train_images.shape[0], height, width, channel))
test_images = test_images.reshape((test_images.shape[0], height, width, channel))
train_images, test_images = train_images / 255.0, test_images / 255.0

model = create_model()
model.summary()


start=time.time()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience = 3, restore_best_weights = True)
model.fit(train_images, train_labels, epochs=8,callbacks=[early_stopping])
end=time.time()


model.save('models/char_recognition_cnn_v2.keras')   
print('Saved model')
print(f'Training duration: {end - start} seconds')



test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)
print('Test loss: ', test_loss)



