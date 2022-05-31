import numpy as np

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model

# Rutas
model_dir = 'model/model.h5'
weights_dir = 'model/weights.h5'

# Parametros
img_height = 100
img_width = 100
class_names = ['Bean', 'Bitter gourd', 'Bottle gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot',
               'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']


model = load_model(model_dir)
model.load_weights(weights_dir)


def predict(file):
    x = load_img(file, target_size=(img_height, img_width))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    arr = model.predict(x)
    result = arr[0]
    response = np.argmax(result)

    if response == 0:
        print(class_names[0])
    elif response == 1:
        print(class_names[1])
    elif response == 2:
        print(class_names[2])
    elif response == 3:
        print(class_names[3])
    elif response == 4:
        print(class_names[4])
    elif response == 5:
        print(class_names[5])
    elif response == 6:
        print(class_names[6])
    elif response == 7:
        print(class_names[7])
    elif response == 8:
        print(class_names[8])
    elif response == 9:
        print(class_names[9])
    elif response == 10:
        print(class_names[10])
    elif response == 11:
        print(class_names[11])
    elif response == 12:
        print(class_names[12])
    elif response == 13:
        print(class_names[13])
    elif response == 14:
        print(class_names[14])

    return response

# Example
predict('Vegetable Images/test/Cauliflower/1048.jpg')
