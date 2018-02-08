from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib

local_device_protos = device_lib.list_local_devices()
for x in local_device_protos:
    print(x.name)

print("---")

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(196, 196, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])

batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='constant',
    cval=255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'data/learning',  # this is the target directory
    target_size=(196, 196),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')


validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(196, 196),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')

model.fit_generator(
        train_generator,
        steps_per_epoch=4800 // batch_size,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save(f'2_2_1_1_2f.h5')
