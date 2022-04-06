# coding=utf-8
import tensorflow as tf
from keras.layers import BatchNormalization
# from keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.applications import xception, vgg19
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from image import reverse_image
from training import training, training2


def create_model(base_model_name, classes, img_size, img_channels):
    if base_model_name == 'xception':
        base_model = xception.Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(img_size[0], img_size[1], img_channels),
            pooling='avg')
    else:
        base_model = vgg19.VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=(img_size[0], img_size[1], img_channels),
            pooling='avg')
    base_model.trainable = False

    top_model = Sequential()
    top_model.add(Dense(512, activation='relu', input_shape=base_model.output_shape[1:]))
    top_model.add(BatchNormalization())
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(BatchNormalization())
    top_model.add(Dense(1024, activation='relu'))
    # top_model.add(Dropout(0.5))
    top_model.add(Dense(len(classes), activation='softmax'))

    training_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=top_model(base_model.output)
    )

    training_model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.01, nesterov=True),  # lr-0.001
        loss='binary_crossentropy',
        metrics='accuracy'
    )
    training_model.summary()

    style_layer_output = training_model.get_layer('block5_conv1').output

    output_model = tf.keras.models.Model(
        inputs=training_model.input,
        outputs=(style_layer_output, training_model.output)
    )

    return training_model, output_model


def load_model(model_name):
    training_model = keras.models.load_model(model_name)

    style_layer_output = training_model.get_layer('block5_conv1').output

    output_model = tf.keras.models.Model(
        inputs=training_model.input,
        outputs=(style_layer_output, training_model.output)
    )

    return training_model, output_model


base_model_name = 'vgg19'

classes = ['cat', 'dog']

img_height = img_width = 224
img_size = (img_width, img_height)
img_channels = 3

# training_model, output_model = create_model(base_model_name, classes, img_size, img_channels)
training_model, output_model = load_model('model8_cats_vs_dogs_vgg19.h5')
training2(training_model, output_model, base_model_name, 'model8_cats_vs_dogs_vgg19.h5', classes, img_size)

result_class_number, image = reverse_image(training_model, base_model_name, 0, img_size, 'dog_9236.png')
plt.imshow(image)
plt.show()
image.save("result.jpg")
