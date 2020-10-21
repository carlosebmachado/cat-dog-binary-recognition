from keras_preprocessing import image
from keras.models import model_from_json
import numpy as np

# 0 cat
# 1 dog
IMG_SIZE = 128


# LOAD
json_file = open('saved_model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights('saved_model/model.h5')


# COMPILE
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])


# TEST
test_img = image.load_img('test/cat/cat04.jpg', target_size=(IMG_SIZE,IMG_SIZE))


test_img = image.load_img('test/dog/dog03.jpg', target_size=(IMG_SIZE,IMG_SIZE))



test_img = image.img_to_array(test_img)
test_img /= 255
test_img = np.expand_dims(test_img, axis=0)
print(classifier.predict(test_img))
