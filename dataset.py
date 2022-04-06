# coding=utf-8
import os
import random
import shutil
from os import listdir

from image import reverse_image


def create_dataset(size, path, model, model_name, epoch):
    if os.path.exists(path + "/train2/"):
        shutil.rmtree(path + "/train2/")
    os.makedirs(path + "/train2/")
    if epoch == 0:
        category_size = size // 2
        fake_size = 1
    else:
        category_size = size // 3
        fake_size = size // 3

    all_cats_filenames = listdir(path + "/cat224/")
    random.shuffle(all_cats_filenames)
    cats_filenames = all_cats_filenames[:category_size]
    all_dogs_filenames = listdir(path + "/dog224/")
    random.shuffle(all_dogs_filenames)
    dogs_filenames = all_dogs_filenames[:category_size]

    count = 0
    for filename in cats_filenames:
        extension = filename.split(".")[-1]
        shutil.copy(path + "/cat224/" + filename, "{}/train2/cat_{}.{}".format(path, count, extension))
        count += 1
    for filename in dogs_filenames:
        extension = filename.split(".")[-1]
        shutil.copy(path + "/dog224/" + filename, "{}/train2/dog_{}.{}".format(path, count, extension))
        count += 1

    all_fakes_filenames = listdir(path + "/fake224/")
    all_fakes_count = len(all_fakes_filenames)
    if epoch > 1:
        if all_fakes_count > fake_size * 2 // 3:
            if all_fakes_count > 5000:
                all_fakes_filenames = all_fakes_filenames[all_fakes_count - 5000:]
            random.shuffle(all_fakes_filenames)
            fake_filenames = all_fakes_filenames[:fake_size * 2 // 3]
            for filename in fake_filenames:
                if ('dog' in filename):
                    animal = 'dog'
                else:
                    animal = 'cat'
                extension = filename.split(".")[-1]
                shutil.copy(path + "/fake224/" + filename, "{}/train2/{}_{}.{}".format(path, animal, count, extension))
                count += 1
                all_fakes_count += 1

    new_fake_filenames = all_cats_filenames[category_size:] + all_dogs_filenames[category_size:]
    random.shuffle(new_fake_filenames)
    for filename in new_fake_filenames:
        category = filename.split('.')[0]
        if 'dog' in category:
            animal = 'dog'
            reverse_category = 0
            filename = path + "/dog224/" + filename
        else:
            animal = 'cat'
            reverse_category = 1
            filename = path + "/cat224/" + filename

        result_category, image = reverse_image(model, model_name, reverse_category, (224, 224), filename)
        if result_category == reverse_category:
            image.save("{}/fake224/{}_{}.png".format(path, animal, all_fakes_count))
            image.save("{}/train2/{}_{}.png".format(path, animal, count))
            count += 1
            all_fakes_count += 1
            if count >= size:
                break
        else:
            print("WARNING failed to create reverse image")

    if count < size:
        print("ERROR failed to create training pack")
