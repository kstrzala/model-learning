import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


def get_outer_model_5(batch_norm=True):
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


def get_outer_model_conv_5():
    return tf.keras.Sequential([
        kl.BatchNormalization(axis=3),
        kl.Conv2D(16, kernel_size=7, strides=3, padding='same', activation='elu'),
        kl.BatchNormalization(axis=3),
        kl.Conv2D(16, kernel_size=3, strides=2, padding='same', activation='elu'),
        kl.BatchNormalization(axis=3),
        kl.Conv2D(4, kernel_size=3, strides=2, padding='same')
    ])

def get_outer_model_conv_10():
    return tf.keras.Sequential([
        kl.Conv2D(16, kernel_size=7, strides=2, padding='same', activation='elu'),
        kl.Conv2D(32, kernel_size=3, strides=1, padding='valid', activation='elu'),
        kl.Conv2D(32, kernel_size=3, strides=1, padding='valid', activation='elu'),
        kl.MaxPool2D(),
        kl.Conv2D(4, kernel_size=3, strides=1, padding='valid', activation='elu'),
        kl.MaxPool2D(),
    ])


def get_inner_model(input_size):
    return tf.keras.Sequential([
        kl.Reshape((input_size,)),
        kl.BatchNormalization(axis=1),
        kl.Dense(50, activation='elu'),
        kl.BatchNormalization(axis=1),
        kl.Dense(4, activation='softmax')
    ])


def get_inner_model_linear(input_size):
    return tf.keras.Sequential([
        kl.Reshape((input_size,)),
        kl.BatchNormalization(axis=1),
        kl.Dense(4),
    ])


def get_inner_model_conv_5(batch_norm=True):
    if batch_norm:
        return tf.keras.Sequential([
            kl.BatchNormalization(axis=3),
            kl.Conv2D(16, 3, activation='elu'),
            kl.MaxPool2D(),
            kl.BatchNormalization(axis=3),
            kl.Conv2D(16, 16, activation='elu'),
            kl.BatchNormalization(axis=3),
            kl.Dense(4)
        ])
    else:
        return tf.keras.Sequential([
            kl.Conv2D(16, 3, activation='elu'),
            kl.MaxPool2D(),
            kl.Conv2D(32, 16, activation='elu'),
            kl.Dense(4, activation='softmax')
        ])


def get_inner_model_conv_10():
    return tf.keras.Sequential([
            kl.BatchNormalization(axis=3),
            kl.Conv2D(8, kernel_size=3, padding='same', activation='elu'),
            kl.Conv2D(16, kernel_size=3, padding='same', activation='elu'),
            kl.MaxPool2D(),
            kl.BatchNormalization(axis=3),
            kl.Conv2D(32, kernel_size=5, padding='valid', activation='elu'),
            kl.Reshape((32,)),
            kl.Dense(4, activation='softmax')
        ])


class FullModel(tf.keras.Model):
    def __init__(self, maze_size):
        super(FullModel, self).__init__()
        if maze_size==5:
            self.OuterModelGenerator = lambda: get_outer_model_5(batch_norm=False)
            self.InnerModelGenerator = lambda: get_inner_model(5*5*4)
        elif maze_size==10:
            self.OuterModelGenerator = get_outer_model_conv_10
            self.InnerModelGenerator = get_inner_model_conv_10
        else:
            raise RuntimeError("Maze size accepted are only 5 and 10")

        self.outer_model = self.OuterModelGenerator()
        self.inner_model = self.InnerModelGenerator()

    def call(self, x, outer_train=False, inner_train=False):
        y = self.outer_model(x, training=outer_train)
        y = self.inner_model(y, training=inner_train)
        return y

    def policy(self, x):
        y = tf.convert_to_tensor(np.expand_dims(x, 0).astype(np.float32))
        y = self(y)
        return y.numpy()[0]

    def reset_inner_model(self):
        self.inner_model = self.InnerModelGenerator()