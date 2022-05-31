import os.path
import tensorflow as tf

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential

# Rutas
path = 'Vegetable Images/'
test_dir = path + 'test'
train_dir = path + 'train'
vali_dir = path + 'validation'

# Parametros
batch_size = 32
img_height = 100
img_width = 100
epochs = 10
steps_per_epoch = 100
validation_steps = 20
filter_size = (3, 2)
filter_size2 = (2, 2)
pool_size = (2, 2)
num_classes = 15
lr = 0.0005

# Pre procesamiento de imágenes
image_data = ImageDataGenerator(rescale=1./255)

image_test_data = image_data.flow_from_directory(
    test_dir, target_size=(img_height, img_width), class_mode='categorical', batch_size=batch_size)
image_train_data = image_data.flow_from_directory(
    train_dir, target_size=(img_height, img_width), class_mode='categorical', batch_size=batch_size, subset='training')
image_vali_data = image_data.flow_from_directory(
    vali_dir, target_size=(img_height, img_width), class_mode='categorical', batch_size=batch_size, subset='validation')

# Creación del modelo
model = Sequential([
    layers.Conv2D(32, filter_size, padding='same', input_shape=(img_height, img_width, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=pool_size),
    layers.Conv2D(64, filter_size2, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=pool_size),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenando al modelo
model.fit(image_train_data, epochs=epochs, validation_data=image_vali_data, steps_per_epoch=steps_per_epoch,
          validation_steps=validation_steps)

if not os.path.exists('model/'):
    os.mkdir('model/')
model.save('model/model.h5')
model.save_weights('model/weights.h5')
