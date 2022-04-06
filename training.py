# coding=utf-8
from os import listdir

import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import xception, vgg19

from dataset import create_dataset


def loadImagesAsDataFrames(path, classes):
    filenames = listdir(path)
    categories = []
    for f_name in filenames:
        category = f_name.split('.')[0]
        for c in classes:
            if c in category:
                categories.append(c)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    return df


def training(model, base_model_name, model_name, classes, img_size, path="./cats-vs-dogs/train/", epochs=2):
    if base_model_name == 'xception':
        preprocessing_function = xception.preprocess_input
    else:
        preprocessing_function = vgg19.preprocess_input

    df = loadImagesAsDataFrames(path, classes)

    train_df = df
    # train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    # train_df = train_df.reset_index(drop=True)
    total_train = train_df.shape[0]
    # validate_df = validate_df.reset_index(drop=True)
    # total_validate = validate_df.shape[0]
    batch_size = 30

    train_datagen = ImageDataGenerator(
        # rotation_range=15,
        preprocessing_function=preprocessing_function,
        # shear_range=0.1,
        # zoom_range=0.1,
        # horizontal_flip=False,
        # width_shift_range=0.1,
        # height_shift_range=0.1
    )
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        path,
        x_col='filename',
        y_col='category',
        target_size=img_size,
        class_mode='categorical',
        batch_size=batch_size
    )

    # validation_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    # validation_generator = validation_datagen.flow_from_dataframe(
    #     validate_df,
    #     path,
    #     x_col='filename',
    #     y_col='category',
    #     target_size=img_size,
    #     class_mode='categorical',
    #     batch_size=batch_size)

    early_stop = EarlyStopping(monitor='loss', min_delta=0.05, patience=5)
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='accuracy',  # val_sequential_accuracy
        patience=4,
        verbose=True,
        factor=0.5,
        min_lr=0.00001)
    callbacks = [early_stop, learning_rate_reduction]

    history = model.fit(
        train_generator,
        epochs=epochs,
        # validation_data=validation_generator,
        # validation_steps=total_validate // batch_size,
        steps_per_epoch=total_train // batch_size,
        callbacks=callbacks
    )

    model.save(model_name)

    # plt.plot(history.history['sequential_accuracy'])
    # plt.show()


def training2(training_model, output_model, base_model_name, model_name, classes, img_size):
    for i in range(250):
        print("global epoch {}".format(i))
        create_dataset(450, "./cats-vs-dogs/", output_model, base_model_name, i)
        training(training_model, base_model_name, model_name, classes, img_size, "./cats-vs-dogs/train2/", 5)
