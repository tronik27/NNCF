from tensorflow.keras.layers import Reshape, Layer, BatchNormalization, Conv2D, Dense, Flatten, Add
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm


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

    def __init__(self, initial_filter_matrix, num_correlations=32):
        super(NNCFModel, self).__init__()
        self.initial_filter_matrix = initial_filter_matrix
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

    def get_correlation_filter(self, plot=False):
        cf = self(self.initial_filter_matrix, training=False)[1]
        if plot:
            plt.imshow(cf[0, :, :, 0], cmap='gray')
            plt.show()
        return cf[0, :, :, 0]

    def plot_output_correlations(self):
        correlations = self(self.initial_filter_matrix, training=False)[0]
        if correlations.shape[-1] % 4:
            correlations = correlations[:, :, :, :-3]
        fig, axes = plt.subplots(nrows=4, ncols=correlations.shape[-1] // 4, figsize=(16, 16))
        i = 0
        for axe in axes.flat:
            axe.imshow(correlations[0, :, :, i], cmap=cm.jet)
            i = i + 1
        plt.show()
