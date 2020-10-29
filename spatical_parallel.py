import tensorflow as tf
from tensorflow import keras
from .vpnn import vpnn
from vpnn.types import Permutation_options
from donwload_util import load_mnist_stash

(x_train, y_train), (x_test, y_test) = load_mnist_stash()

x_train = x_train.reshape(-1, 28*28) / 255
x_test = x_test.reshape(-1, 28*28) / 255
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


input_one = keras.Input(28*28, name='input_one_in')
# input_two = keras.Input(28*28, name='input_two_in')


x1 = vpnn(input_dim=28*28, n_layers=4, n_rotations=10,
          permutation_arrangement=Permutation_options.horizontal,
          hidden_activation="chebyshev",
          M_init=2.0,
          name='input_one')(input_one)

x1 = keras.layers.BatchNormalization()(x1)
x2 = keras.layers.BatchNormalization()(x2)

x2 = vpnn(input_dim=28*28, n_layers=4, n_rotations=10,
          permutation_arrangement=Permutation_options.vertical,
          hidden_activation="chebyshev",
          M_init=2.0,
          name='input_two')(input_one)

concat = keras.layers.concatenate([x1, x2])
output = keras.layers.Dense(10, activation='softmax', name='output')(concat)

model = keras.Model(input_one, output)
model.compile('adam',
              loss='categorical_crossentropy', metrics='accuracy')
model.summary()

model.fit(x_train, y_train,
          batch_size=128,
          epochs=100,
          validation_data=(x_test, y_test)
          )
