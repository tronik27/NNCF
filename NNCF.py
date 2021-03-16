from tensorflow.keras.layers import Input, Reshape, Layer, BatchNormalization, Conv2D, Dense, Flatten, Add, experimental
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from Correlation_utils import Plot3D
from nncf_utils import data_prepare


class ResidualBlock(Layer):

    def __init__(self, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        This method should build the layers according to the above specification. Make sure
        to use the input_shape argument to get the correct number of filters, and to set the
        input_shape of the first layer in the block.
        """
        self.BatchNorm_1 = BatchNormalization(input_shape=input_shape[1:2])
        self.conv_1 = Conv2D(input_shape[-1], (3, 3), activation=None, padding='same')
        self.BatchNorm_2 = BatchNormalization()
        self.conv_2 = Conv2D(input_shape[-1], (3, 3), activation=None, padding='same')

    def call(self, inputs, training=False):
        """
        This method should contain the code for calling the layer according to the above
        specification, using the layer objects set up in the build method.
        """
        x = self.BatchNorm_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1(x)
        x = self.BatchNorm_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        return Add()([x, inputs])


class FiltersChangeResidualBlock(Layer):

    def __init__(self, out_filters, **kwargs):
        """
        The class initialiser should call the base class initialiser, passing any keyword
        arguments along. It should also set the number of filters as a class attribute.
        """
        super(FiltersChangeResidualBlock, self).__init__(**kwargs)
        self.out_filters = out_filters

    def build(self, input_shape):
        """
        This method should build the layers according to the above specification. Make sure
        to use the input_shape argument to get the correct number of filters, and to set the
        input_shape of the first layer in the block.
        """
        self.BatchNorm_1 = BatchNormalization(input_shape=input_shape[1:2])
        self.conv_1 = Conv2D(input_shape[-1], (3, 3), activation=None, padding='same')
        self.BatchNorm_2 = BatchNormalization()
        self.conv_2 = Conv2D(self.out_filters, (3, 3), activation=None, padding='same')
        self.conv_3 = Conv2D(self.out_filters, (1, 1), activation=None)

    def call(self, inputs, training=False):
        """
        This method should contain the code for calling the layer according to the above
        specification, using the layer objects set up in the build method.
        """
        x = self.BatchNorm_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1(x)
        x = self.BatchNorm_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x1 = self.conv_3(inputs)
        return Add()([x, x1])


def create_model(shape, num_correlations, gt_lable, sample_weight, initial_filter_matrix):
    inputs = Input(shape=shape[1:])
    conv_1 = Conv2D(32, (7, 7), activation='relu', strides=(2, 2))(inputs)
    residual_block = ResidualBlock()(conv_1)
    conv_2 = Conv2D(32, (3, 3), activation='relu', strides=(2, 2))(residual_block)
    filters_change_residual_block = FiltersChangeResidualBlock(64)(conv_2)
    flatten = Flatten()(filters_change_residual_block)
    dense = Dense(shape[1] ** 2, activation='sigmoid')(flatten)
    cf = Reshape(shape[1:], name='corr_filter')(dense)
    outputs = Conv2D(filters=num_correlations, kernel_size=shape[1], use_bias=False,
                     padding='same', activation=None, trainable=False, name='correlation')(cf)

    model = NNCFModel(initial_filter_matrix, gt_lable=gt_lable, sample_weight=sample_weight)(inputs, [outputs, cf])

    return model


class NNCFModel(Model):

    def __init__(self, initial_filter_matrix, num_correlations=32, gt_lable=0, sample_weight=(1, 100)):
        super(NNCFModel, self).__init__()
        self.gt_lable = gt_lable
        self.initial_filter_matrix = initial_filter_matrix
        self.sample_weight = sample_weight
        self.num_correlations = num_correlations
        self.conv_1 = Conv2D(32, (7, 7), activation='relu', strides=(2, 2))
        self.residual_block = ResidualBlock()
        self.conv_2 = Conv2D(32, (3, 3), activation='relu', strides=(2, 2))
        self.filters_change_residual_block = FiltersChangeResidualBlock(64)
        self.flatten = Flatten()
        self.dense = Dense(self.initial_filter_matrix.shape[1] ** 2, activation='sigmoid')
        self.reshape = Reshape(self.initial_filter_matrix.shape[1:], name='corr_filter')
        self.correlation = Conv2D(filters=self.num_correlations, kernel_size=self.initial_filter_matrix.shape[1],
                                  use_bias=False, padding='same', activation=None, trainable=False, name='correlation')

    def call(self, inputs):
        conv_1 = self.conv_1(inputs)
        residual_block = self.residual_block(conv_1)
        conv_2 = self.conv_2(residual_block)
        filters_change_residual_block = self.filters_change_residual_block(conv_2)
        flatten = self.flatten(filters_change_residual_block)
        dense = self.dense(flatten)
        cf = self.reshape(dense)
        outputs = self.correlation(cf)
        return outputs, cf

    def get_correlation_filter(self):
        cf = self(self.initial_filter_matrix, training=False)[1]
        return cf

    def train_step(self, data):
        _, ground_truth, sample_weights = data

        with tf.GradientTape() as tape:
            y_pred = self(self.initial_filter_matrix, training=True)[0]
            loss = self.compiled_loss(ground_truth, y_pred, sample_weight=sample_weights,
                                      regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(ground_truth, y_pred, sample_weight=sample_weights)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        _, ground_truth, sample_weights = data
        y_pred = self(self.initial_filter_matrix, training=False)[0]
        self.compiled_loss(ground_truth, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(ground_truth, y_pred)

        return {m.name: m.result() for m in self.metrics}


class SetWeightsCallback(tf.keras.callbacks.Callback):

    def __init__(self, generator):
        super(SetWeightsCallback, self).__init__()
        self.generator = generator

    def on_train_batch_begin(self, batch, logs=None):
        weights, _, _ = self.generator[batch]
        weights = np.expand_dims(weights, 4)
        weights = np.transpose(weights, [3, 1, 2, 4, 0])
        self.model.get_layer('correlation').set_weights(weights)

    def on_test_batch_begin(self, batch, logs=None):
        weights, _, _ = self.generator[batch]
        weights = np.expand_dims(weights, 4)
        weights = np.transpose(weights, [3, 1, 2, 4, 0])
        self.model.get_layer('correlation').set_weights(weights)


num_correlations = 32
epochs = 10
num_of_images = 3000
initial_filter_matrix = tf.zeros((1, 28, 28, 1))
gt_lable = 0
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_data, validation_data, shape = data_prepare(train_images, train_labels, test_images, test_labels,
                                                  label=0,
                                                  num_of_images=num_of_images,
                                                  num_of_corr=num_correlations)

nncf = NNCFModel(num_correlations=num_correlations,
                 gt_lable=gt_lable,
                 sample_weight=(1, 100),
                 initial_filter_matrix=initial_filter_matrix)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
nncf.build(input_shape=(32, 28, 28, 1))
nncf.compile(optimizer=tf.keras.optimizers.Adam(0.002),
             loss=tf.keras.losses.MeanSquaredError(),
             metrics=['accuracy'])
print('wsh1:', shape)

print(nncf.summary())
history = nncf.fit(train_data, steps_per_epoch=num_of_images // num_correlations,
                   callbacks=[SetWeightsCallback(train_data)], epochs=epochs)
