import numpy as np
import random as rand
import math
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf

def gcu(x):

    return tf.math.multiply(x, tf.math.cos(x))
class GenomeHandler:


    def __init__(self, max_dense_layers, max_filters,
                 max_dense_nodes, input_shape, n_classes,
                 batch_normalization=True, dropout=True, max_pooling=True,
                 optimizers=None, activations=None):
      
        if max_dense_layers < 1:
            raise ValueError(
                "At least one dense layer is required for softmax layer"
            )
        if max_filters > 0:
            filter_range_max = int(math.log(max_filters, 2)) + 1
        else:
            filter_range_max = 0
        self.optimizer = optimizers or [
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta',

        ]
        self.activation = activations or [
            'relu',
            'sigmoid',
            'swish',
            'gelu',
            gcu,
        ]

        self.dense_layer_shape = [
            "active",
            "num nodes",
            "batch normalization",
            "activation",
            "dropout",
        ]
        self.layer_params = {
            "active": [0, 1],
            "num filters": [2**i for i in range(3, filter_range_max)],
            "num nodes": [2**i for i in range(4, int(math.log(max_dense_nodes, 2)) + 1)],
            "batch normalization": [0, (1 if batch_normalization else 0)],
            "activation": list(range(len(self.activation))),
            "dropout": [(i if dropout else 0) for i in range(11)],
            "max pooling": list(range(3)) if max_pooling else 0,
        }

        self.dense_layers = max_dense_layers - 1
        self.dense_layer_size = len(self.dense_layer_shape)
        self.input_shape = input_shape
        self.n_classes = n_classes



    def denseParam(self, i):
        key = self.dense_layer_shape[i]
        return self.layer_params[key]

    def mutate(self, genome, num_mutations):
        num_mutations = np.random.choice(num_mutations)
        for i in range(num_mutations):
            index = np.random.choice(list(range(1, len(genome))))
            offset = self.dense_layer_size * self.dense_layers

            new_index = (index - offset)
            present_index = new_index - new_index % self.dense_layer_size
            if genome[present_index + offset]:
                range_index = new_index % self.dense_layer_size
                choice_range = self.denseParam(range_index)
                genome[index] = np.random.choice(choice_range)
            elif rand.uniform(0, 1) <= 0.01:
                genome[present_index + offset] = 1
            else:
                genome[index] = np.random.choice(list(range(len(self.optimizer))))
        return genome

    def decode(self, genome):
        if not self.is_compatible_genome(genome):
            raise ValueError("Invalid genome for specified configs")
        model = Sequential()
        dim = 0
        offset = 0


        for i in range(self.dense_layers):
            if genome[offset]:
                dense = None

                dense = Dense(genome[offset + 1], input_shape=self.input_shape)
                input_layer = False
                else:
                    dense = Dense(genome[offset + 1])
                model.add(dense)
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                model.add(Activation(self.activation[genome[offset + 3]]))
                model.add(Dropout(float(genome[offset + 4] / 20.0)))
            offset += self.dense_layer_size

        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer[genome[offset]],
                      metrics=["accuracy"])
        return model

    def genome_representation(self):
        encoding = []

        for i in range(self.dense_layers):
            for key in self.dense_layer_shape:
                encoding.append("Dense" + str(i) + " " + key)
        encoding.append("Optimizer")
        return encoding

    def generate(self):
        genome = []

        for i in range(self.dense_layers):
            for key in self.dense_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))
        genome.append(np.random.choice(list(range(len(self.optimizer)))))
        genome[0] = 1
        return genome

    def is_compatible_genome(self, genome):

        ind = 0
        for i in range(self.dense_layers):
            for j in range(self.dense_layer_size):
                if genome[ind + j] not in self.denseParam(j):
                    return False
            ind += self.dense_layer_size
        if genome[ind] not in range(len(self.optimizer)):
            return False
        return True

    def best_genome(self, csv_path, metric="accuracy", include_metrics=True):
        best = max if metric is "accuracy" else min
        col = -1 if metric is "accuracy" else -2
        data = np.genfromtxt(csv_path, delimiter=",")
        row = list(data[:, col]).index(best(data[:, col]))
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        return genome

    def decode_best(self, csv_path, metric="accuracy"):
        return self.decode(self.best_genome(csv_path, metric, False))
