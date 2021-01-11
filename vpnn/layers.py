import tensorflow as tf
import numpy as np
from random import choice

from math import sqrt
from .types import Permutation_options


def update_slice(arr, startx, endx, starty, endy, new_arr):
    for i in range(startx, endx):
        for j in range(starty, endy):
            arr[i, j] = new_arr[i-startx, j-starty]


def gen_grid_permutation(width, height, max_range=10, offset=None):
    used_offset = offset or max_range

    # numbers from 0 to dim-1
    perm = np.arange(width*height).reshape(width, height)

    for i in range(0, height, used_offset):
        for j in range(0, width, used_offset):
            if(i+max_range > height or j+max_range > width):
                continue
            # get the current slice
            part = perm[i:i+max_range, j:j+max_range].flatten()
            # shuffle it
            np.random.shuffle(part)
            part = part.reshape(max_range, max_range)
            # print("part")
            # print(part)
            # update it in the permuation
            update_slice(perm, i, i+max_range, j, j+max_range, part)
            # print(perm)

    return perm.flatten()


def gen_horizontal_permutation_row(dim, max_range=10, n=None):
    # take a number and put it in the top 5 available spots
    # prem = [-1 for i in range(dim)]
    perm = []

    # numbers from 0 to dim-1
    numbers = [i for i in range(dim)] if n is None else n

    currentMax = np.array(numbers[:max_range])

    for i in range(dim):

        #  shuffle the current top
        temp = np.array(currentMax)
        np.random.shuffle(temp)
        currentMax = temp.tolist()

        perm.append(currentMax.pop(0))

        if(i + max_range < dim):
            currentMax.append(numbers[i+max_range])
    return perm


def gen_horizontal_permutation(dim, height, width, max_range=10):
    perm = []

    for i in range(height):
        def add_row(x: int):
            return i*width + x

        new_row = map(add_row, gen_horizontal_permutation_row(
            width, max_range=max_range))
        perm.extend(new_row)
    return np.array(perm)


def gen_vertical_permutaton(dim, height, width, max_range=10):
    nums = [[] for i in range(height)]

    # make a matrix of the numbers 1,2,3,4,..dim
    count = 0
    for i in range(height):
        nums[i] = [j for j in range(count, count+width)]
        count = count + width
    nums_np = np.array(nums)
    perm = []
    for i in range(width):
        perm.append(gen_horizontal_permutation_row(
            height, max_range=max_range, n=nums_np[:, i:i+1].flatten()))
    return np.array(perm).T.flatten()


class Permutation(tf.keras.layers.Layer):
    def __init__(self, perm=None, permutation_arrangement=Permutation_options.random, max_range=10, ** kwargs):
        super().__init__(**kwargs)
        self.permutation_arrangement = permutation_arrangement
        self.permutation = perm
        self.max_range = max_range

    def get_config(self):
        conf = super().get_config()
        conf.update({'perm': self.permutation})
        return conf

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        dim = input_shape[-1]
        width = int(sqrt(dim))
        if self.permutation:
            pass
        elif not self.permutation and self.permutation_arrangement == Permutation_options.random:
            self.permutation = np.random.permutation(dim)

        if self.permutation_arrangement == Permutation_options.horizontal:
            self.permutation = gen_horizontal_permutation(
                dim,
                width,
                width,
                max_range=self.max_range
            )
        elif self.permutation_arrangement == Permutation_options.vertical:
            self.permutation = gen_vertical_permutaton(
                dim,
                width,
                width,
                max_range=self.max_range
            )
        elif self.permutation_arrangement == Permutation_options.mixed:
            num = choice([1, 2])
            if num == 1:
                self.permutation = gen_vertical_permutaton(dim,
                                                           width,
                                                           width,
                                                           max_range=self.max_range)
            else:
                self.permutation = gen_horizontal_permutation(dim,
                                                              width,
                                                              width,
                                                              max_range=self.max_range)
        elif self.permutation_arrangement == Permutation_options.grid:
            self.permutation = gen_grid_permutation(
                width, width, max_range=self.max_range)
        elif self.permutation_arrangement == Permutation_options.mixed3:
            num = choice([1, 2, 3])
            if num == 1:
                self.permutation = gen_vertical_permutaton(dim,
                                                           width,
                                                           width,
                                                           max_range=self.max_range)
            elif num == 2:
                self.permutation = gen_horizontal_permutation(dim,
                                                              width,
                                                              width,
                                                              max_range=self.max_range)
            else:
                self.permutation = gen_grid_permutation(
                    width, width, max_range=self.max_range)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.gather(inputs, self.permutation, axis=-1)


class Rotation(tf.keras.layers.Layer):
    def __init__(self, theta_initializer='uniform', **kwargs):
        super().__init__(**kwargs)
        self.units = None
        self.theta = None
        self.theta_initializer = theta_initializer

    def get_config(self):
        conf = super().get_config()
        conf.update({'theta_initializer': self.theta_initializer})
        return conf

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.units = input_shape[-1]
        if self.units % 2 == 1:
            raise ValueError(
                'Rotation layer only works on an even number of inputs')
        self.theta = self.add_weight(name='theta',
                                     initializer=self.theta_initializer,
                                     shape=(self.units//2,))
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        cos = tf.cos(self.theta)
        sin = tf.sin(self.theta)
        xi = inputs[..., ::2]
        xj = inputs[..., 1::2]
        yi = cos * xi - sin * xj
        yj = cos * xj + sin * xi
        return tf.reshape(tf.stack([yi, yj], axis=-1), tf.shape(inputs))


class Diagonal(tf.keras.layers.Layer):
    def __init__(self, t_initializer='uniform', function=None, **kwargs):
        super().__init__(**kwargs)
        self.units = None
        self.t = None
        self.t_initializer = t_initializer
        self.function = function or tf.nn.sigmoid

    def get_config(self):
        conf = super().get_config()
        conf.update({'t_initializer': self.t_initializer,
                     'function': self.function})
        return conf

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.units = input_shape[-1]
        self.t = self.add_weight(name='t',
                                 initializer=self.t_initializer,
                                 shape=(self.units,))
        # sample uniformity between 0 and 0.1
        # there default was 0.01
        # self.m = self.add_weight(name='m',  initializer=)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # f = M*f(t/M) + M
        # M could be a vector
        f = self.function(self.t)
        # element wise div on the vector f and roll of F
        vec = f / tf.roll(f, -1, 0)
        return inputs * vec


class Bias(tf.keras.layers.Layer):
    def __init__(self, bias_initializer='uniform', **kwargs):
        super().__init__(**kwargs)
        self.bias = None
        self.units = None
        self.bias_initializer = bias_initializer

    def get_config(self):
        conf = super().get_config()
        conf.update({'bias_initializer': self.bias_initializer})
        return conf

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.units = input_shape[-1]
        self.bias = self.add_weight(name='bias',
                                    initializer=self.bias_initializer,
                                    shape=(self.units,))
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.bias


class SVDDownsize(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.Z = None
        self.units = units
        super().__init__(**kwargs)

    def get_config(self):
        conf = super().get_config()
        conf.update({'units': self.units})
        return conf

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def build(self, input_shape):
        v = np.random.normal(size=(input_shape[-1], self.units))
        z = np.linalg.svd(v, full_matrices=False)[0]
        self.Z = tf.Variable(z, dtype=tf.float32, trainable=False)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.Z)


class KernelWrapper(tf.keras.layers.Layer):
    """
    A wrapper for keras layers that, for a layer L
    and input x of dimension n, computes dot(L(I_n), x)
    instead of L(x). Mostly for vpnns, where the kernel V
    is not implemented as a matrix but may be desired to sometimes
    be treated as such.
    """

    def __init__(self, layer, clip_args=None, **kwargs):
        """
        creates a kernel wrapper
        :param layer: the layer object to wrap
        :param clip_args: arguments passed to tf.clip_by_value if not None
        :param kwargs: passed to super constructor
        """
        self.layer = layer
        self.clip_args = clip_args
        self.units = None
        super().__init__(**kwargs)

    def get_config(self):
        """
        implements the method of the super class + the layer and clip args
        """
        conf = super().get_config()
        conf.update({'layer': self.layer, 'clip_args': self.clip_args})
        return conf

    def compute_output_shape(self, input_shape):
        """
        computes the output shape of the model
        :param input_shape: input shape to the model
        :return: the result of the wrapped layer calling the same method
        """
        return self.layer.compute_output_shape(input_shape)

    def build(self, input_shape):
        """
        builds the wrapped layer
        :param input_shape: input shape for wrapped layer
        """
        self.layer.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        computes the transformation dot(inputs, self.layer(eye(n)))
        along with appropriate element-wise clipping
        :param inputs: inputs to the model
        :param kwargs: unused
        :return: a tensor representing the result
        """
        kernel = self.layer(tf.eye(tf.shape(inputs)[-1]))
        if self.clip_args is not None:
            kernel = tf.clip_by_value(kernel, *self.clip_args)
        return tf.matmul(inputs, kernel)
