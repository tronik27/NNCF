import tensorflow as tf
import numpy as np
import datetime
import shutil
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os


class FilterDataset:

    def __init__(self, gt_label=0, num_of_corr=32, num_of_images=None, augmentation=None, sample_weight=(1, 100, 200)):
        self.gt_label = gt_label
        self.num_of_corr = num_of_corr
        self.num_of_images = num_of_images
        self.augmentation = augmentation
        self.sample_weights = sample_weight

    def prepare_data_from_array(self, train_images, train_labels, validation_images=None, validation_labels=None,
                                num_of_images_last=False):

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
        if num_of_images_last:
            validation_images = np.transpose(validation_images)
            train_images = np.transpose(train_images)

        if not self.num_of_images or self.num_of_images > train_images.shape[0]:
            self.num_of_images = train_images.shape[0]
            print('num_of_images set to {}!'.format(self.num_of_images))

        train_images, train_labels = self._balance_class(images=train_images, labels=train_labels)
        print('num_of_images set to {}!'.format(train_images.shape[-1]))
        train_weights = self._make_data_generator(images=train_images, labels=train_labels)

        self.augmentation = None
        validation_images, validation_labels = self._balance_class(validation_images, validation_labels)
        validation_weights = self._make_data_generator(images=validation_images,
                                                       labels=validation_labels)

        return train_weights, validation_weights, train_images.shape

    def prepare_data_from_directory(self, train_path, validation_path=None, train_labels_path=None,
                                    validation_labels_path=None, target_size=(100, 100)):
        if train_labels_path:
            train_image_names, train_labels, = self.get_data_from_csv(train_labels_path, train_path)
            if validation_labels_path:
                validation_image_names, validation_labels = self.get_data_from_csv(validation_labels_path,
                                                                                   validation_path)
        else:
            train_image_names, train_labels = self.get_data_from_directory(train_path)
            if validation_path:
                validation_image_names, validation_labels = self.get_data_from_directory(validation_path)

        if not validation_path:
            train_image_names, validation_image_names, train_labels, validation_labels = train_test_split(
                                                                                                    train_image_names,
                                                                                                    train_labels,
                                                                                                    shuffle=True,
                                                                                                    test_size=0.33,
                                                                                                    random_state=42
                                                                                                          )
        if not self.num_of_images or self.num_of_images > len(train_image_names):
            self.num_of_images = len(train_image_names)
            print('num_of_images set to {}!'.format(self.num_of_images))

        train_image_names, train_labels = self._balance_class(images=train_image_names, labels=train_labels)

        train_weights = self._make_data_generator_from_directory(images=train_image_names,
                                                                 labels=train_labels,
                                                                 target_size=target_size)

        self.augmentation = None
        validation_image_names, validation_labels = self._balance_class(images=validation_image_names,
                                                                        labels=validation_labels)
        validation_weights = self._make_data_generator_from_directory(images=validation_image_names,
                                                                      labels=validation_labels,
                                                                      target_size=target_size)

        return train_weights, validation_weights

    @staticmethod
    def get_data_from_csv(labels_path, path):
        df = pd.read_csv(labels_path)
        file_names = df.iloc[:, 0].tolist()
        labels = df.iloc[:, 1].to_numpy()
        images_path = list(map(lambda x: path + '/' + x, file_names))
        return images_path, labels

    @staticmethod
    def get_data_from_directory(path):
        labels = []
        images_path = []
        for label, directory in enumerate(list(map(lambda x: path + '/' + x, os.listdir(path)))):
            images = list(map(lambda x: directory + '/' + x, os.listdir(directory)))
            images_path = images_path + images
            labels = labels + list(label * np.ones(len(images)))
        return images_path, np.array(labels)

    def _balance_class(self, images, labels, test_mode=False):
        if not isinstance(images, list):
            if len(images.shape) == 3:
                images = np.expand_dims(images, 3)
            elif images.shape[3] > 1:
                images = np.mean(images, axis=3, keepdims=True)
        else:
            images = np.array(images)

        positive_images = images[labels == self.gt_label]
        positive_labels = labels[labels == self.gt_label]
        negative_images = images[labels != self.gt_label]
        negative_labels = labels[labels != self.gt_label]
        if self.num_of_images // 2 < len(positive_images):
            if isinstance(images[0], str):
                positive_images = positive_images[:self.num_of_images // 2]
                negative_images = negative_images[:self.num_of_images // 2]
            else:
                positive_images = positive_images[:self.num_of_images // 2, :, :, :]
                negative_images = negative_images[:self.num_of_images // 2, :, :, :]
            positive_labels = positive_labels[:self.num_of_images // 2]
            negative_labels = negative_labels[:self.num_of_images // 2]
        else:
            if isinstance(images[0], str):
                negative_images = negative_images[:positive_images.shape[0]]
            else:
                negative_images = negative_images[:positive_images.shape[0], :, :, :]
            negative_labels = negative_labels[:positive_images.shape[0]]

        if isinstance(images[0], str):
            images = np.hstack((positive_images, negative_images))
        else:
            images = np.vstack((positive_images, negative_images))
        labels = np.hstack((positive_labels, negative_labels))
        if not test_mode:
            if len(images) % self.num_of_corr:
                if isinstance(images[0], str):
                    images = images[:len(images) - len(images) % self.num_of_corr]
                else:
                    images = images[:images.shape[0] - images.shape[0] % self.num_of_corr, :, :, :]
                labels = labels[:len(images) - len(images) % self.num_of_corr]

        labels = labels == self.gt_label
        labels = labels.astype(int)
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

    def _make_data_generator_from_directory(self, images, labels, target_size):
        gt, sample_weight = self._make_gt_correlation(shape=(len(images), target_size[0], target_size[1], 1),
                                                      labels=labels)
        print(gt.shape, sample_weight.shape)
        datagen = CustomDataGen(images_path=images,
                                labels=gt,
                                batch_size=self.num_of_corr,
                                input_size=(target_size[0], target_size[1], 1),
                                sample_weight=sample_weight,
                                shuffle=True,
                                is_train=True)
        return datagen

    def _make_gt_correlation(self, shape, labels):
        gt = np.zeros(shape, dtype='float32')
        class_weights = np.ones(shape, dtype='float32')*self.sample_weights[0]

        if shape[1] % 2:
            x1, x2 = shape[1] // 2 - 1, shape[1] // 2 + 2
        else:
            x1, x2 = shape[1] // 2 - 1, shape[1] // 2 + 1

        gt[np.where(labels == self.gt_label), x1:x2, x1:x2, :] = 1
        class_weights[np.where(labels != self.gt_label), :, :, :] = 1 * self.sample_weights[1]
        class_weights[np.where(labels == self.gt_label), x1:x2, x1:x2, :] = 1 * self.sample_weights[2]
        return gt, class_weights

    def make_test_data_from_array(self, images, labels):
        images, labels = self._balance_class(images=images, labels=labels, test_mode=True)
        return images, labels

    def make_test_data_from_directory(self, path):
        pass


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, images_path, labels, batch_size, sample_weight=None,
                 input_size=(256, 256, 1), shuffle=True, is_train=True):
        self.images_path = images_path
        self.labels = labels
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.is_train = is_train
        self.number_of_samples = len(self.labels)
        self.sample_weights = sample_weight

    def on_epoch_end(self):
        if self.is_train:
            shuffler = np.random.permutation(len(self.images_path))
            self.images_path = self.images_path[shuffler]
            self.labels = self.labels[shuffler]
            if self.sample_weights:
                self.sample_weights = self.sample_weights[shuffler]

    def __getitem__(self, index):
        batch_images_path = self.images_path[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size, :, :, :]
        batch_sample_weights = self.sample_weights[index * self.batch_size:(index + 1) * self.batch_size, :, :, :]
        batch_images = np.zeros((self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2]))
        for i, path in enumerate(batch_images_path):
            batch_images[i, :, :, :] = self._get_image(path)

        return np.float32(batch_images), batch_labels, batch_sample_weights

    def __len__(self):
        return self.number_of_samples // self.batch_size

    def _get_image(self, path):

        image = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale', target_size=(self.input_size[0],
                                                                                                 self.input_size[1]))
        image = tf.keras.preprocessing.image.img_to_array(image)

        return image / 255.


class SetWeightsCallback(tf.keras.callbacks.Callback):

    def __init__(self, generator):
        super(SetWeightsCallback, self).__init__()
        self.generator = generator

    def on_train_batch_begin(self, batch, logs=None):
        weights, _, _ = self.generator[batch]
        #weights = tf.math.subtract(weights, tf.reduce_mean(weights, axis=(1, 2), keepdims=True))
        weights = np.expand_dims(weights, 4)
        weights = np.transpose(weights, [3, 1, 2, 4, 0])
        self.model.get_layer('correlation').set_weights(weights)

    def on_test_batch_begin(self, batch, logs=None):
        weights, _, _ = self.generator[batch]
        #weights = tf.math.subtract(weights, tf.reduce_mean(weights, axis=(1, 2), keepdims=True))
        weights = np.expand_dims(weights, 4)
        weights = np.transpose(weights, [3, 1, 2, 4, 0])
        self.model.get_layer('correlation').set_weights(weights)


class CFMetric(tf.keras.metrics.Metric):

    def __init__(self, shape, test_mode=False, name='accuracy', peak_classifier='pce', **kwargs):
        super(CFMetric, self).__init__(name=name, **kwargs)
        metrics = {'auc': tf.keras.metrics.SpecificityAtSensitivity(0.5),
                   'precision': tf.keras.metrics.Precision(),
                   'recall': tf.keras.metrics.Recall(),
                   'accuracy': tf.keras.metrics.BinaryAccuracy()}
        peak_classifiers = {'pce': self.pce, 'peak_position': self.peak_position}
        self.metric_value = self.add_weight(name=name, initializer='zeros')
        self.shape = shape
        self.test_mode = test_mode
        self.metric = metrics[name]
        self.peak_classifier = peak_classifiers[peak_classifier]

    def update_state(self, gt_correlation, correlation, sample_weight=None):
        y_true, y_pred = self.correlation_classification(gt_correlation, correlation)

        if self.name in ['precision', 'recall', 'auc']:
            y_true, y_pred = y_true[:, 0], y_pred[:, 0]

        self.metric.update_state(y_true, y_pred)
        self.metric_value.assign(self.metric.result())

    def result(self):
        return self.metric_value

    def reset_states(self):
        self.metric_value.assign(0.)

    def correlation_classification(self, gt_correlation, correlation):
        if not self.test_mode:
            labels = gt_correlation[0, self.shape[1]//2, self.shape[2]//2, :]
            correlation = correlation[0, :, :, :]
            correlation = tf.transpose(correlation, perm=[2, 0, 1])
        else:
            labels = gt_correlation

        preds = self.peak_classifier(correlation)
        labels = tf.cast(tf.expand_dims(labels, axis=1), self.dtype)
        preds = tf.cast(tf.expand_dims(preds, axis=1), self.dtype)
        return labels, preds

    def pce(self, scene):
        correlation_energy = tf.math.reduce_sum(scene, axis=(1, 2))
        if self.shape[1] % 2:
            x1, x2 = self.shape[1] // 2 - 3, self.shape[1] // 2 + 4
        else:
            x1, x2 = self.shape[1] // 2 - 3, self.shape[1] // 2 + 3

        peak_energy = tf.math.reduce_sum(scene[:, x1:x2, x1:x2], axis=(1, 2))
        return tf.round(7.05 * tf.math.divide(peak_energy, correlation_energy))

    def peak_position(self, scenes):
        if self.shape[1] % 2:
            x1, x2 = self.shape[1] // 2 - 1, self.shape[1] // 2 + 2
        else:
            x1, x2 = self.shape[1] // 2 - 1, self.shape[1] // 2 + 1
        class_labels = []
        for scene in scenes:
            peak_coordinates = np.unravel_index(scene.argmax(), scene.shape)
            if x1 <= peak_coordinates[0] <= x2 and x1 <= peak_coordinates[1] <= x2:
                class_labels.append(1)
            else:
                class_labels.append(0)
        return np.array(class_labels)


def make_tensorboard():
    path = "logs/fit/"
    if os.path.exists(path):
        shutil.rmtree(path)
    log_dir = path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard
