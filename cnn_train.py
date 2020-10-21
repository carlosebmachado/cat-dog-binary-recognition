from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import json

create = False
IMG_SIZE = 128
BATCH_SIZE = 2


# se não quiser gerar mais imagens, colocar somente o rescale
training_gen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=7,
                                  horizontal_flip=True,
                                  shear_range=0.2,
                                  height_shift_range=0.07,
                                  zoom_range=0.2)
test_gen = ImageDataGenerator(rescale=1./255)

training_base = training_gen.flow_from_directory('dataset/training_set',
                                                 target_size=(IMG_SIZE,IMG_SIZE),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='binary')
test_base = test_gen.flow_from_directory('dataset/test_set',
                                         target_size=(IMG_SIZE,IMG_SIZE),
                                         batch_size=BATCH_SIZE,
                                         class_mode='binary')


if create:
    # save class indices
    print(training_base.class_indices)
    with open('saved_model/class_indices.json', 'w') as json_file:
        json.dump(training_base.class_indices, json_file, indent=4)
    
    
    classifier = Sequential()
    # conv layers
    for i in range(2):
        #filters recomendado: a partir de 64
        classifier.add(Conv2D(32, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 3), activation = 'relu'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten())
    
    # dense layers
    for i in range(2):
        classifier.add(Dense(units=128, activation='relu'))
        classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, activation='sigmoid'))
else:
    # LOAD
    json_file = open('saved_model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    classifier.load_weights('saved_model/model.h5')


# loss='categorical_crossentropy' se houver mais de 2 classes
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])


classifier.fit_generator(training_base,
                         steps_per_epoch=4000 / BATCH_SIZE,
                         epochs=5,
                         validation_data=test_base,
                         validation_steps=1000 / BATCH_SIZE)


model_json = classifier.to_json()
with open('saved_model/model.json', 'w') as json_file:
    json_file.write(model_json)
classifier.save_weights('saved_model/model.h5')
