from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
from keras.models import model_from_json
import numpy as np


IMG_SIZE = 128
BATCH_SIZE = 2

class CDRModel:
    
    classifier = None
    
    training_base = None
    test_base = None
    
    
    def __init__(self):
        pass
    
    
    def create(self):
        self.classifier = Sequential()
        # conv layers
        for i in range(2):
            #filters recomendado: a partir de 64
            self.classifier.add(Conv2D(64, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 3), activation = 'relu'))
            self.classifier.add(BatchNormalization())
            self.classifier.add(MaxPooling2D(pool_size=(2,2)))
        self.classifier.add(Flatten())
        
        # dense layers
        for i in range(2):
            self.classifier.add(Dense(units=128, activation='relu'))
            self.classifier.add(Dropout(0.2))
        self.classifier.add(Dense(units=1, activation='sigmoid'))
    
    
    def compile(self):
        # loss='categorical_crossentropy' se houver mais de 2 classes
        self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    
    
    def load_data(self):
        # se não quiser gerar mais imagens, colocar somente o rescale
        training_gen = ImageDataGenerator(rescale=1./255,
                                          rotation_range=7,
                                          horizontal_flip=True,
                                          shear_range=0.2,
                                          height_shift_range=0.07,
                                          zoom_range=0.2)
        test_gen = ImageDataGenerator(rescale=1./255)
        
        self.training_base = training_gen.flow_from_directory('dataset/training_set',
                                                         target_size=(IMG_SIZE,IMG_SIZE),
                                                         batch_size=BATCH_SIZE,
                                                         class_mode='binary')
        self.test_base = test_gen.flow_from_directory('dataset/test_set',
                                                         target_size=(IMG_SIZE,IMG_SIZE),
                                                         batch_size=BATCH_SIZE,
                                                         class_mode='binary')
        print(self.training_base.class_indices)
    
    
    def load(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.classifier = model_from_json(loaded_model_json)
        self.classifier.load_weights("model.h5")
        
    
    
    def save(self):
        model_json = self.classifier.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        self.classifier.save_weights("model.h5")
    
    
    def train(self, times):
        self.load_data()
        self.classifier.fit_generator(self.training_base,
                                      steps_per_epoch=4000 / BATCH_SIZE,
                                      epochs=times,
                                      validation_data=self.test_base,
                                      validation_steps=1000 / BATCH_SIZE)
    
    
    def recognize(self, path):
        test_img = image.load_img(path, target_size=(IMG_SIZE,IMG_SIZE))
        test_img = image.img_to_array(test_img)
        test_img /= 255
        test_img = np.expand_dims(test_img, axis=0)
        return self.classifier.predict(test_img)
    
    
    def get_accuracy(self):
        return self.classifier.evaluate(self.test_predictors, self.test_class)
