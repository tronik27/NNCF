import tensorflow as tf
import numpy as np
import datetime
import os
import shutil
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class FilterDataset:

    def __init__(self, gt_label=0, num_of_corr=32, num_of_images=None, augmentation=None, target_size=(100, 100),
                 sample_weight=(1, 200), num_of_images_last=False):
        self.gt_label = gt_label
        self.num_of_corr = num_of_corr
        self.num_of_images = num_of_images
        self.augmentation = augmentation
        self.target_size = target_size
        self.sample_weights = sample_weight
        self.num_of_images_last = num_of_images_last

    def prepare_data_from_array(self, train_images, train_labels, validation_images=None, validation_labels=None):

        if not validation_images.any():
            train_images, train_labels, validation_images, validation_labels = train_test_split(train_images,
                                                                                                train_labels,
                                                                                                shuffle=True,
                                                                                                test_size=0.33,
                                                                                                random_state=42)

        if not (3 <= len(train_images.shape) <= 4 or 3 <= len(validation_images.shape) <= 4):
            raise ValueError('Train data shape should be 3 or 4 dimensional! Got {}-dimensional data array!'.format(
                                                                                                 len(train_images.shape)
            ))
        if self.num_of_images_last:
            validation_images = np.transpose(validation_images)
            train_images = np.transpose(train_images)

        if not self.num_of_images or self.num_of_images > train_images.shape[0]:
            self.num_of_images = train_images.shape[0]
            print('num_of_images set to {}!'.format(self.num_of_images))

        train_images, train_labels = self._balance_class(images=train_images, labels=train_labels)
        print(train_images.shape)
        train_weights = self._make_data_generator(images=train_images, labels=train_labels)

        self.augmentation = None
        validation_images, validation_labels = self._balance_class(validation_images, validation_labels)
        validation_weights = self._make_data_generator(images=validation_images,
                                                       labels=validation_labels)

        return train_weights, validation_weights, train_images.shape

    def _balance_class(self, images, labels):
        if len(images.shape) == 3:
            images = np.expand_dims(images, 3)
        elif images.shape[3] > 1:
            images = np.mean(images, axis=3, keepdims=True)
        print(images.shape)
        positive_images = images[labels == self.gt_label]
        positive_labels = labels[labels == self.gt_label]
        negative_images = images[labels != self.gt_label]
        negative_labels = labels[labels != self.gt_label]
        if self.num_of_images // 2 < positive_images.shape[0]:
            positive_images = positive_images[:self.num_of_images // 2, :, :, :]
            negative_images = negative_images[:self.num_of_images // 2, :, :, :]
            positive_labels = positive_labels[:self.num_of_images // 2]
            negative_labels = negative_labels[:self.num_of_images // 2]
        else:
            negative_images = negative_images[:self.num_of_images - positive_images.shape[0], :, :, :]
            negative_labels = negative_labels[:self.num_of_images - positive_images.shape[0]]
        images = np.vstack((positive_images, negative_images))
        labels = np.hstack((positive_labels, negative_labels))
        if images.shape[0] % self.num_of_corr:
            images = images[:images.shape[0] - images.shape[0] % self.num_of_corr, :, :, :]
            labels = labels[:images.shape[0] - images.shape[0] % self.num_of_corr]
        print(images.shape)
        return images, labels

    def _make_data_generator(self, images, labels):
        gt, sample_weight = self._make_gt_correlation(shape=images.shape, labels=labels)
        if self.augmentation:
            rotation_range, horizontal_flip, vertical_flip = self.augmentation
            datagen = ImageDataGenerator(rescale=1 / 255.,
                                         rotation_range=rotation_range,
                                         horizontal_flip=horizontal_flip,
                                         vertical_flip=vertical_flip)
        else:
            datagen = ImageDataGenerator(rescale=1 / 255.)

        data = datagen.flow(images, gt, sample_weight=sample_weight, seed=42, batch_size=self.num_of_corr)
        return data

    def _make_gt_correlation(self, shape, labels):
        gt = np.zeros(shape, dtype='float32')
        class_weights = np.ones(shape, dtype='float32') * self.sample_weights[0]

        if shape[1] % 2:
            x1, x2 = shape[1] // 2 - 1, shape[1] // 2 + 2
            gt[np.where(labels == self.gt_label), x1:x2, x1:x2, :] = 1
            class_weights[np.where(labels == self.gt_label), x1:x2, x1:x2, :] = 1 * self.sample_weights[1]
        else:
            x1, x2 = shape[1] // 2 - 1, shape[1] // 2 + 1
            gt[np.where(labels == self.gt_label), x1:x2, x1:x2, :] = 1
            class_weights[np.where(labels == self.gt_label), x1:x2, x1:x2, :] = 1 * self.sample_weights[1]
        return gt, class_weights

    def prepare_data_from_directory(self, train_path, validation_path=None):
        pass


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


def make_tensorboard():
    path = "logs/fit/"
    if os.path.exists(path):
        shutil.rmtree(path)
    log_dir = path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard
