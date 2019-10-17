import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                      origin=_URL,
                                      extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 2

    real_image = image[:, :w, :]
    facade_image = image[:, w:, :]

    real_image = tf.cast(real_image, tf.float32)
    facade_image = tf.cast(facade_image, tf.float32)

    return real_image, facade_image


def resize(real_image, facade_image, width, height):
    real_image = tf.image.resize(real_image, [width, height], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    facade_image = tf.image.resize(facade_image, [width, height], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return real_image, facade_image


def random_crop(real_image, facade_image):
    stacked_image = tf.stack([real_image, facade_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


def normalize(real_image, facade_image):
    real_image = (real_image / 127.5) - 1
    facade_image = (facade_image / 127.5) - 1

    return real_image, facade_image


def random_jitter(real_image, facade_image):
    real_image, facade_image = resize(real_image, facade_image, 290, 290)
    real_image, facade_image = random_crop(real_image, facade_image)

    if np.random.uniform(0, 1, 1) > 0.5:
        real_image = tf.image.flip_left_right(real_image)
        facade_image = tf.image.flip_left_right(facade_image)

    return real_image, facade_image


def load_train_image(image_file):
    rel, fac = load(image_file)
    rel, fac = random_jitter(rel, fac)
    rel, fac = normalize(rel, fac)
    return rel, fac


def load_test_image(image_file):
    rel, fac = load(image_file)
    rel, fac = resize(rel, fac, IMG_WIDTH, IMG_HEIGHT)
    rel, fac = normalize(rel, fac)
    return rel, fac


train_data = tf.data.Dataset.list_files(PATH+"train/*.jpg")
train_data = train_data.map(map_func=load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_data = train_data.cache().shuffle(BATCH_SIZE)
train_data = train_data.batch(1)

test_data = tf.data.Dataset.list_files(PATH+'test/*.jpg')
test_data = test_data.map(load_test_image)
test_data = test_data.batch(1)


def un_sample(filters, size, use_normalization=True):
    initializer = tf.random_normal_initializer(0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=[1, 2, 2, 1], padding="SAME"
                               , kernel_initializer=initializer)
    )

    if use_normalization:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def up_sample(filters, size, use_normalization=True, use_dropout = True, rate = 0.5):
    initializer = tf.random_normal_initializer(0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=[1, 2, 2, 1], padding="SAME"
                                        , kernel_initializer=initializer)
    )

    if use_normalization:
        result.add(tf.keras.layers.BatchNormalization())

    if use_dropout:
        result.add(tf.keras.layers.Dropout(rate=rate))

    result.add(tf.keras.layers.LeakyReLU())

    return result




with tf.Session() as sess:
    re, inp = load(PATH + "train/100.jpg")
    re, inp = random_jitter(re, inp)
    re = re.eval()
    inp = inp.eval()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(re / 255.0)
    plt.subplot(1, 2, 2)
    plt.imshow(inp / 255.0)
    plt.show()
