# coding=utf-8
import numpy as np
import tensorflow as tf
from PIL import Image as PImage
from keras.preprocessing.image import image
from tensorflow.keras.applications import xception, vgg19


def deprocess_img_xception(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
        assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                   "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessing step
    x[:, :, 0] = (x[:, :, 0] + 1) * 128
    x[:, :, 1] = (x[:, :, 1] + 1) * 128
    x[:, :, 2] = (x[:, :, 2] + 1) * 128
    # x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def deprocess_img_vgg19(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))


def compute_loss(model, loss_weights, init_image, complete_similarity, target_class_number):
    model_outputs = model(init_image)

    # similarity = model_outputs[0][0]
    similarity = [style_layer[0] for style_layer in model_outputs[:1]]
    probability_score = model_outputs[1][0][target_class_number]

    # similarity_score = get_content_loss(similarity, complete_similarity)
    similarity_score = 0
    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 # / float(5)
    for comb_style, target_style in zip(similarity, complete_similarity):
        similarity_score += weight_per_style_layer * get_style_loss(comb_style, target_style)

    loss = similarity_score * loss_weights[0] + (1 - probability_score) * loss_weights[1]
    # print('probability_score ' + str(probability_score.numpy()))
    # print('loss ' + str(loss.numpy()))
    # print('---------------------------')
    return loss, similarity_score, probability_score


def reverse_image(model, model_name, target_class_number, img_size, image_name):
    if model_name == 'vgg19':
        preprocessing_function = vgg19.preprocess_input
    else:
        preprocessing_function = xception.preprocess_input

    img = image.load_img(image_name, target_size=img_size)
    x_img = preprocessing_function(np.expand_dims(image.img_to_array(img), axis=0))

    num_iterations = 10

    model_outputs = model(x_img)

    complete_similarity = [gram_matrix(style_feature[0]) for style_feature in model_outputs[:1]]

    init_image = np.copy(x_img)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=2, beta1=0.5, epsilon=1e-3)
    best_loss, best_img = float('inf'), None
    loss_weights = (0.001, 1000)

    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'complete_similarity': complete_similarity,
        'target_class_number': target_class_number
    }

    min_vals = -1
    max_vals = 1
    if model_name == 'vgg19':
        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            all_loss = compute_loss(**cfg)

        loss, similarity_score, probability_score = all_loss
        grads = tape.gradient(loss, init_image)

        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

        if loss < best_loss:
            best_loss = loss
            best_img = init_image

        if loss < loss_weights[1] / 5:
            break

    model_outputs = model(best_img)
    result_class_number = np.argmax(model_outputs[1][0].numpy())

    print('beast loss ' + str(best_loss.numpy()))

    if best_loss > loss_weights[1] / 3:
        return int(1 != target_class_number), best_img

    if model_name == 'xception':
        best_img = deprocess_img_xception(best_img.numpy())
    else:
        best_img = deprocess_img_vgg19(best_img.numpy())

    best_image = PImage.fromarray(best_img.astype('uint8'), 'RGB')

    return result_class_number, best_image
