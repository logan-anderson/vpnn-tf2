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

parser.add_argument('--permutation_arrangement', type=int, default=1,
                    help='random = 1  horizontal = 2 vertical = 3 mixed = 4')
parser.add_argument('--use_dropout', type=bool, default=False, help='')
parser.add_argument('--total_runs', type=int, default=28,
                    help='it will run tests from 1 to total_runs')
parser.add_argument('--epochs', type=int, default=2,
                    help='total epochs on each test')

args = parser.parse_args()
print(args)
n_layers = args.layers
n_rotations = args.rotations
total = args.total_runs
dropout = args.use_dropout
total_epochs = args.epochs
# data prep
(x_train, y_train), (x_test, y_test) = load_mnist_stash()

x_train = x_train.reshape(-1, 28*28) / 255
x_test = x_test.reshape(-1, 28*28) / 255
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

(x_train, y_train), (x_val, y_val) = (
    x_train[:-5000], y_train[:-5000]), (x_train[-5000:], y_train[-5000:])


def build_model(max_):
    model = vpnn(input_dim=28*28, n_layers=n_layers, n_rotations=n_rotations,
                 hidden_activation="chebyshev",
                 M_init=2.0,
                 output_dim=10,
                 name=f'vpnn-1',
                 output_activation='softmax',
                 permutation_arrangement=types.Permutation_options.mixed,
                 max_permution_range=max_
                 )

    model.compile(optimizer='adam',
                  metrics='accuracy', loss='categorical_crossentropy')
    print(f'max_range={max_}')
    model.summary()
    return model


validations = []
stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=10, restore_best_weights=True)

for i in range(total):
    model = build_model(i)
    hist = model.fit(x_train, y_train, epochs=total_epochs,
                     validation_data=(x_val, y_val), callbacks=[stopping_callback])
    current_max = max(hist.history['val_accuracy'])
    print('hist', hist.history)
    print('max', current_max)
    value = model.evaluate(x_test, y_test)
    print(value)
    validations.append(value[-1])

fig, ax = plt.subplots()
ax.plot([i + 1 for i in range(total)],
        validations, label='Mixed Permutations')
ax.plot([i + 1 for i in range(total)], [.92]
        * total, label='random permutations')
ax.set(xlabel="max range", ylabel="Test Accuracy",
       title=f"layers={n_layers} rotations={n_rotations}, mixed permutations",
       label='mixed permutations'
       )
ax.legend()

now = datetime.now()
fig.savefig(
    f'./img/plot_layers={n_layers}_rotations={n_rotations}_mixed permutations-{now.strftime("%Y-%m-%d %H:%M:%S")}.png')
print('all done!')
