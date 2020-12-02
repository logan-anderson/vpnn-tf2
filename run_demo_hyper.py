import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import mod
from vpnn import vpnn, types
import argparse
from matplotlib import pyplot as plt
from datetime import datetime


from donwload_util import load_mnist_stash


# args
parser = argparse.ArgumentParser()

parser.add_argument('--layers', type=int, default=1,
                    help='number of vpnn layers in each part')

parser.add_argument('--rotations', type=int, default=2,
                    help='number of vpnn layers in each part')

parser.add_argument('--output_dim', type=int, default=None,
                    help='number of vpnn layers in each part')
parser.add_argument('-fake', action='store_true')


args = parser.parse_args()
print(args)
n_layers = args.layers
n_rotations = args.rotations
output_dim = args.output_dim
total = 3
dropout = True

# data prep
(x_train, y_train), (x_test, y_test) = load_mnist_stash()

x_train = x_train.reshape(-1, 28*28) / 255
x_test = x_test.reshape(-1, 28*28) / 255

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


def build_model(max_):
    model = vpnn(input_dim=28*28, n_layers=n_layers, n_rotations=n_rotations,
                 hidden_activation="chebyshev",
                 M_init=2.0,
                 output_dim=10,
                 name=f'vpnn-1',
                 output_activation='softmax',
                 permutation_arrangement=types.Permutation_options.mixed,
                 #  max_permution_range=hp.Int('units',
                 #                             min_value=1,
                 #                             max_value=5,
                 #                             step=1)
                 max_permution_range=max_
                 )

    model.compile(optimizer='adam',
                  metrics='accuracy', loss='categorical_crossentropy')
    print(f'max_range={max_}')
    model.summary()
    return model


validations = []
for i in range(total):
    model = build_model(total)
    hist = model.fit(x_train, y_train, epochs=2,
                     validation_data=(x_test, y_test))
    current_max = max(hist.history['val_accuracy'])
    print('hist', hist.history)
    print('max', current_max)
    validations.append(current_max)

fig, ax = plt.subplots()
ax.plot([i + 1 for i in range(total)], validations)
ax.set(xlabel="max range", ylabel="validation accuracy",
       title=f"layers={n_layers} rotations={n_rotations}, mixed permutations")

now = datetime.now()
fig.savefig(
    f'./img/plot_layers={n_layers}_rotations={n_rotations}_mixed permutations-{now.strftime("%Y-%m-%d %H:%M:%S")}.png')
print('all done!')
