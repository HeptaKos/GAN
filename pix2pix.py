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
ALPHA=100


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


@tf.function()
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


def down_sample(filters, size, use_normalization=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=[2, 2], padding="SAME"
                               , kernel_initializer=initializer, use_bias=False)
    )

    if use_normalization:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def up_sample(filters, size, use_normalization=True, use_dropout = False, rate = 0.5):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=[2, 2], padding="SAME"
                                        , kernel_initializer=initializer, use_bias=False)
    )

    if use_normalization:
        result.add(tf.keras.layers.BatchNormalization())

    if use_dropout:
        result.add(tf.keras.layers.Dropout(rate=rate))

    result.add(tf.keras.layers.LeakyReLU())

    return result


def generator():
    down = [
        down_sample(64, 4, use_normalization=False),
        down_sample(128, 4),
        down_sample(256, 4),
        down_sample(512, 4),
        down_sample(512, 4),
        down_sample(512, 4),
        down_sample(512, 4),
        down_sample(512, 4),
    ]

    up = [
        up_sample(512, 4, use_dropout=True),
        up_sample(512, 4, use_dropout=True),
        up_sample(512, 4, use_dropout=True),
        up_sample(512, 4),
        up_sample(256, 4),
        up_sample(128, 4),
        up_sample(64, 4),   # 128 128 64
    ]

    concate = tf.keras.layers.Concatenate()
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(3, 4, strides=[2, 2], padding="SAME"
                                        , kernel_initializer=initializer, activation="tanh", use_bias=False)
    )

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs

    mark = []

    for layers in down:
        x = layers(x)
        mark.append(x)

    mark = reversed(mark[:-1])

    for layers, marks in zip(up, mark):
        x = layers(x)
        x = concate([x, marks])

    x = result(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    target = tf.keras.layers.Input(shape=[None, None, 3], name="target_image")
    inputs = tf.keras.layers.Input(shape=[None, None, 3], name="inputs_image")

    x = tf.keras.layers.concatenate([inputs, target])

    down1 = down_sample(64, 4, use_normalization=False)(x)
    down2 = down_sample(128, 4)(down1)
    down3 = down_sample(256, 4)(down2)  # 32 32 256

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)

    conv1 = tf.keras.layers.Conv2D(512, 4, strides=[1, 1]
                                   , kernel_initializer=initializer
                                   , use_bias=False)(zero_pad1)

    batch1 = tf.keras.layers.BatchNormalization()(conv1)

    relu1 = tf.keras.layers.LeakyReLU()(batch1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(relu1)

    conv2 = tf.keras.layers.Conv2D(1, 4, strides=[1, 1]
                                   , kernel_initializer=initializer
                                   , use_bias=False)(zero_pad2)

    return tf.keras.Model(inputs=[inputs, target], outputs=conv2)



loss_method = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(dis_gens_output, dis_real_output):
    loss1 = loss_method(tf.zeros_like(dis_gens_output), dis_gens_output)
    loss2 = loss_method(tf.ones_like(dis_real_output), dis_real_output)
    total_loss = loss1+loss2
    return total_loss


def generator_loss(dis_gens_output, gens_output, targe_image):
    loss1 = loss_method(tf.ones_like(dis_gens_output), dis_gens_output)
    loss2 = tf.reduce_mean(tf.abs(gens_output-targe_image))
    loss = loss1+ALPHA*loss2
    return loss


generator_sample = generator()
discriminator_sample = discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


def train(input_image, target):
    with tf.GradientTape() as gen_gra, tf.GradientTape() as dis_gra:
        gen_output = generator_sample(input_image, training=True)
        dis_real_output = discriminator_sample([input_image, target], training=True)
        dis_gen_output = discriminator_sample([input_image, gen_output], training=True)

        gen_loss = generator_loss(dis_gen_output, gen_output, target)
        dis_loss = discriminator_loss(dis_gen_output, dis_real_output)
        #print("gen_loss is : ", gen_loss., "  dis_loss is : ", dis_loss)

    gen_gradient = gen_gra.gradient(gen_loss, generator_sample.trainable_variables)
    dis_gradient = dis_gra.gradient(dis_loss, discriminator_sample.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_gradient, generator_sample.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(dis_gradient, discriminator_sample.trainable_variables))


def fit(datas, epochs):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        count = 0
        for epoch in range(epochs):
            for input_images, target_images in datas:
                train(input_images, target_images)
                print("count  ", count, "  done")
                count = count+1
            print("epoch : ", epoch)


if __name__ == '__main__':
    fit(train_data, 50)