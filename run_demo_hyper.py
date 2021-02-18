from args import CommandLineArgs
import tensorflow as tf
from vpnn import vpnn, types
import argparse
from matplotlib import pyplot as plt
from datetime import datetime
import json

from donwload_util import load_mnist_fasion_stash, load_mnist_stash

PrintCallback = tf.keras.callbacks.LambdaCallback(
    on_epoch_begin=None,  on_epoch_end=lambda epoch, logs: print(f"epoch: {epoch}, accuracy:{logs['accuracy']} loss: {logs['loss']}, val_loss: {logs['val_loss']} val_accuracy: {logs['val_accuracy']}" + '\n'), on_batch_begin=None, on_batch_end=None,
    on_train_begin=None, on_train_end=None,
)
stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=20, restore_best_weights=True)
# args

args = CommandLineArgs()
print(args.args)
perm_type = types.Permutation_options(args.permutation_arrangement)
n_layers = args.layers
n_rotations = args.rotations
total = args.total_runs
dropout = args.dropout
total_epochs = args.epochs
# data prep
(x_train, y_train), (x_test, y_test) = load_mnist_fasion_stash()

x_train = x_train.reshape(-1, 28*28) / 255
x_test = x_test.reshape(-1, 28*28) / 255
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

(x_train, y_train), (x_val, y_val) = (
    x_train[:-5000], y_train[:-5000]), (x_train[-5000:], y_train[-5000:])

perm_type_text = {
    1: 'Random',
    2: 'Horizontal',
    3: 'Vertical',
    4: 'Mixed Horizontal and Vertical',
    5: 'Grid',
    6: 'Mixed Horizontal, Vertical and Grid',
}


def build_model(max_, perm=perm_type):
    model = vpnn(input_dim=28*28, n_layers=n_layers, n_rotations=n_rotations,
                 hidden_activation="chebyshev",
                 M_init=2.0,
                 output_dim=10,
                 name=f'vpnn-{i}',
                 output_activation='softmax',
                 permutation_arrangement=perm,
                 max_permution_range=max_
                 )

    model.compile(optimizer='adam',
                  metrics='accuracy', loss='categorical_crossentropy')
    print(f'max_range={max_}')
    model.summary()
    return model


validations = []


for i in range(total):
    # Needs to be i + 1 as we are starting at 0
    model = build_model(i+1)
    hist = model.fit(x_train, y_train,
                     epochs=total_epochs,
                     validation_data=(x_val, y_val),
                     callbacks=[stopping_callback, PrintCallback],
                     verbose=0,
                     )
    current_max = max(hist.history['val_accuracy'])
    print('hist', hist.history)
    print('max', current_max)
    value = model.evaluate(x_test, y_test, verbose=0)
    print(value)
    validations.append(value[-1])


# build random perm model
model = build_model(1, perm=types.Permutation_options.random)
hist = model.fit(x_train, y_train,
                 epochs=total_epochs,
                 validation_data=(x_val, y_val),
                 callbacks=[stopping_callback, PrintCallback],
                 verbose=0,
                 )
random_acc = model.evaluate(x_test, y_test, verbose=0)[-1]

perm_text = perm_type_text[args.permutation_arrangement]

fig, ax = plt.subplots()
ax.plot([i + 1 for i in range(total)],
        validations, label=f'{perm_text}  Permutations')
ax.plot([i + 1 for i in range(total)], [random_acc]*total,
        label='Random Permutations')
ax.set(xlabel="max range", ylabel="Test Accuracy",
       title=f"Fashion MNIST, Layers={n_layers} Rotations={n_rotations}, {perm_text} Permutations",
       label=f'{perm_text}  Permutations'
       )

ax.legend()
plt.xticks([i+1 for i in range(total)])

now = datetime.now()


# save data
file_name = f'./img/FASHION-plot_layers = {n_layers}_rotations = {n_rotations}_{perm_text.replace(" ","_")}_permutations-{now.strftime("%Y-%m-%d %H:%M:%S")}'
fig.savefig(f"{file_name}.png")
with open(f'{file_name}.json', 'w') as outfile:
    json.dump({
        'random_perm': random_acc,
        'perm': validations,
    }, outfile)
print('all done!')
