import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


def get_outer_model(batch_norm=True):
    if batch_norm:
        return tf.keras.Sequential([
            kl.BatchNormalization(axis=3),
            kl.Conv2D(32, kernel_size=7, strides=3, padding='same', activation='elu'),
            kl.BatchNormalization(axis=3),
            kl.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='elu'),
            kl.BatchNormalization(axis=3),
            kl.Conv2D(100, kernel_size=3, strides=2, padding='same', activation='elu'),
            kl.BatchNormalization(axis=3),
            kl.Conv2D(100, kernel_size=3),
            kl.BatchNormalization(axis=3),
            kl.Conv2D(100, kernel_size=3)
        ])
    else:
        return tf.keras.Sequential([
            kl.Conv2D(32, kernel_size=7, strides=3, padding='same', activation='elu'),
            kl.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='elu'),
            kl.Conv2D(100, kernel_size=3, strides=2, padding='same', activation='elu'),
            kl.Conv2D(100, kernel_size=3),
            kl.Conv2D(100, kernel_size=3)
        ])


def get_outer_model_conv():
    return tf.keras.Sequential([
        kl.BatchNormalization(axis=3),
        kl.Conv2D(16, kernel_size=7, strides=3, padding='same', activation='elu'),
        kl.BatchNormalization(axis=3),
        kl.Conv2D(16, kernel_size=3, strides=2, padding='same', activation='elu'),
        kl.BatchNormalization(axis=3),
        kl.Conv2D(4, kernel_size=3, strides=2, padding='same')
    ])


def get_inner_model():
    return tf.keras.Sequential([
        kl.Reshape((100,)),
        kl.BatchNormalization(axis=1),
        kl.Dense(50, activation='elu'),
        kl.BatchNormalization(axis=1),
        kl.Dense(4, activation='softmax')
    ])


def get_inner_model_linear():
    return tf.keras.Sequential([
        kl.Reshape((100,)),
        kl.BatchNormalization(axis=1),
        kl.Dense(4),
    ])


def get_inner_model_conv(batch_norm=True):
    if batch_norm:
        return tf.keras.Sequential([
            kl.BatchNormalization(axis=3),
            kl.Conv2D(16, 3, activation='elu'),
            kl.BatchNormalization(axis=3),
            kl.Conv2D(16, 16, activation='elu'),
            kl.BatchNormalization(axis=3),
            kl.Dense(4)
        ])
    else:
        return tf.keras.Sequential([
            kl.Conv2D(16, 3, activation='elu'),
            kl.Conv2D(16, 16, activation='elu'),
            kl.Dense(4)
        ])



class FullModel(tf.keras.Model):
    INNER_MODEL = lambda: get_inner_model(batch_norm=False)
    OUTER_MODEL = lambda: get_outer_model(batch_norm=False)

    def __init__(self):
        super(FullModel, self).__init__()
        self.outer_model = FullModel.OUTER_MODEL()
        self.inner_model = FullModel.INNER_MODEL()

    def call(self, x, outer_train=False, inner_train=False):
        y = self.outer_model(x, training=outer_train)
        y = self.inner_model(y, training=inner_train)
        return y

    def policy(self, x):
        y = tf.convert_to_tensor(np.expand_dims(x, 0).astype(np.float32))
        y = self(y)
        return y.numpy()[0]

    def reset_inner_model(self):
        self.inner_model = FullModel.INNER_MODEL()