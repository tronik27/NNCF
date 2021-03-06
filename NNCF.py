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


class NNCFModel(Model):
    def train_step(self, data):
        if len(data) == 3:
            ground_truth, weights, sample_weight = data
        else:
            sample_weight = None
            ground_truth, weights = data

        x = tf.zeros((1, 28, 28, 1))
        print('wsh1:', weights.shape)
        weights = tf.transpose(weights)
        print('wsh2:', weights.shape)
        ground_truth = tf.transpose(ground_truth)[:1, :, :, :]
        print('rwsh:', np.shape(self.get_layer('correlation').weights))
        self.get_layer('correlation').set_weights(weights)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)[0]
            loss = self.compiled_loss(ground_truth, y_pred, sample_weight=sample_weight,
                                      regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(ground_truth, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        ground_truth, weights = data
        x = tf.zeros((1, 28, 28, 1))
        weights = np.transpose(weights)
        ground_truth = np.transpose(ground_truth)[:1, :, :, :]
        self.get_layer('correlation').set_weights(weights)

        y_pred = self(x, training=False)[0]
        self.compiled_loss(ground_truth, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(ground_truth, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def get_correlation_filter(self):
        x = tf.zeros((1, 28, 28, 1))
        cf = self(x, training=False)[1]
        return cf


def create_model(shape, num_correlations):
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

    model = NNCFModel(inputs=inputs, outputs=[outputs, cf])

    return model


num_correlations = 32
epochs = 10
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_data, validation_data, shape = data_prepare(train_images, train_labels, test_images, test_labels,
                                                  label=0,
                                                  num_of_corr=num_correlations)
print(train_data.element_spec)
nncf = create_model(shape=shape, num_correlations=num_correlations)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)

nncf.compile(optimizer=tf.keras.optimizers.Adam(0.002),
             loss=tf.keras.losses.MeanSquaredError(),
             metrics=['accuracy'])

print(nncf.summary())
history = nncf.fit(train_data, validation_data=validation_data, epochs=epochs)
