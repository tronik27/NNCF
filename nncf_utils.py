from tensorflow.keras.layers import Input, Reshape, Layer, BatchNormalization, Conv2D, Dense, Flatten, Add, experimental
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from Correlation_utils import Plot3D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def balance_class(images, lables, lable, num_of_images):
    positive_images = images[lables == lable]
    positive_lables = lables[lables == lable]
    negative_images = images[lables != lable]
    negative_lables = lables[lables != lable]
    if num_of_images//2 < positive_images.shape[0]:
        positive_images = positive_images[:num_of_images//2, :, :]
        negative_images = negative_images[:num_of_images//2, :, :]
        positive_lables = positive_lables[:num_of_images // 2]
        negative_lables = negative_lables[:num_of_images // 2]
    else:
        negative_images = negative_images[:num_of_images - positive_images.shape[0], :, :]
        negative_lables = negative_lables[:num_of_images - positive_images.shape[0]]
    images = np.vstack((positive_images, negative_images))
    lables = np.hstack((positive_lables, negative_lables))
    return images, lables


def make_gt_correlation(shape, labels, gt_lable=0, sample_weight=(1, 100)):
    gt = np.zeros(shape, dtype='float32')
    class_weights = np.ones(shape, dtype='float32')*sample_weight[0]

    if shape[1] % 2:
        x1, x2 = shape[1] // 2 - 1, shape[1] // 2 + 2
        gt[np.where(labels == gt_lable), x1:x2, x1:x2, :] = 1
        class_weights[np.where(labels == gt_lable), x1:x2, x1:x2, :] = 1*sample_weight[1]
    else:
        x1, x2 = shape[1]//2-1, shape[1]//2+1
        gt[np.where(labels == gt_lable), x1:x2, x1:x2, :] = 1
        class_weights[np.where(labels == gt_lable), x1:x2, x1:x2, :] = 1*sample_weight[1]
    return gt, class_weights


def make_data_generator(images, batch_size, augmentation, labels, target_size=None):
    # if not target_size:
    #     target_size = images.shape[1:]
    images = np.expand_dims(images, 3).astype(np.float32)
    gt, sample_weights = make_gt_correlation(shape=images.shape, labels=labels)
    if augmentation:
        rotation_range, horizontal_flip, vertical_flip = augmentation
        datagen = ImageDataGenerator(rescale=1/255.,
                                     rotation_range=rotation_range,
                                     horizontal_flip=horizontal_flip,
                                     vertical_flip=vertical_flip)
    else:
        datagen = ImageDataGenerator(rescale=1 / 255.)

    data = datagen.flow(images, gt, sample_weight=sample_weights, seed=42, batch_size=batch_size, shuffle=True)
    return data


def data_prepare(train_images, train_labels, validation_images=None, validation_labels=None, label=0, num_of_corr=32,
                 num_of_images=None, augmentation=None, target_size=None):

    if not validation_images.any():
        train_images, train_labels, validation_images, validation_labels = train_test_split(train_images,
                                                                                            train_labels,
                                                                                            shuffle=True,
                                                                                            test_size=0.33,
                                                                                            random_state=42)
    if not num_of_images or num_of_images > train_images.shape[0]:
        num_of_images = train_images.shape[0]
        print('num_of_images set to {}!'.format(train_images.shape[0]))

    train_images, train_labels = balance_class(train_images, train_labels, label, num_of_images)
    train_weights = make_data_generator(images=train_images,
                                        batch_size=num_of_corr,
                                        augmentation=augmentation,
                                        labels=train_labels,
                                        target_size=target_size)

    validation_images, validation_labels = balance_class(validation_images, validation_labels, label, num_of_images)
    validation_weights = make_data_generator(images=validation_images,
                                             batch_size=num_of_corr,
                                             augmentation=augmentation,
                                             labels=validation_labels,
                                             target_size=target_size)

    return train_weights, validation_weights, train_images.shape


def PrepareSVHS():

    train = loadmat(r'D:\MIFI\SCIENTIFIC WORK\DATASETS\SVHN dataset\train_32x32.mat')
    test = loadmat(r'D:\MIFI\SCIENTIFIC WORK\DATASETS\SVHN dataset\train_32x32.mat')
    train, labels_train = train['X'][:, :, :, :], train['y'][:]
    test, labels_test = test['X'][:, :, :, :], test['y'][:]
    train = np.transpose(train, axes=[3, 0, 1, 2])
    test = np.transpose(test, axes=[3, 0, 1, 2])
    train = train[labels_train[:, 0] == 5]
    print(train.shape)
    return train, test, labels_train, labels_test


def filter_classes(dataset, classes):
    """
    This function should filter the dataset by only retaining dataset elements whose
    label belongs to one of the integers in the classes list.
    The function should then return the filtered Dataset object.
    """
    dataset = dataset.filter(lambda images, labels: tf.reduce_any(tf.equal(labels,tf.constant(classes, dtype=tf.int64))))
    return dataset


def mse_loss(pred, target):
    loss = tf.math.reduce_mean(tf.square(target - pred), axis=3, keepdims=True)
    return loss


def mean_loss(loss):
    sum = tf.reduce_sum(loss, axis=(1, 2))
    return tf.reduce_mean(sum)


def fit(model, num_epochs, train_dataset, validation_dataset, x, start_matrix, optimizer, loss, average_loss, grad_fn):

    train_loss_results = []
    batch_loss_results = []
    train_accuracy_results = []
    acc = tf.keras.metrics.Accuracy()

    for epoch in range(num_epochs):
        epoch_loss_results = []
        for (weights, ground_truth) in train_dataset:
            weights = np.transpose(weights)
            ground_truth = np.transpose(ground_truth)[:1, :, :, :]
            model.get_layer('correlation').set_weights(weights)
            loss_value, grads = grad_fn(model, start_matrix, ground_truth, loss)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_mean = average_loss(loss_value)
            acc.update_state(ground_truth, model(start_matrix)[0])
            epoch_loss_results.append(loss_mean.numpy())

        cf = model(x)[1]
        train_loss_results.append(np.array(epoch_loss_results).mean())
        train_accuracy_results.append(acc.result().numpy())
        batch_loss_results = batch_loss_results + epoch_loss_results
        print('Epoch {}: Loss: {} Accuracy: {}'.format(epoch, np.array(epoch_loss_results).mean(), acc.result().numpy()))
    return train_loss_results, train_accuracy_results, batch_loss_results, cf


def learning_curves_plot(train_loss, train_accuracy):
    fig, axes = plt.subplots(1, 2, sharex=False, figsize=(18, 5))

    axes[0].set_xlabel("Epochs", fontsize=14)
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].set_title('Loss vs epochs')
    axes[0].plot(train_loss)

    axes[1].set_title('Accuracy vs epochs')
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epochs", fontsize=14)
    axes[1].plot(train_accuracy)
    plt.show()


@tf.function
def grad(model, inputs, targets, loss):
    """
    This function should compute the loss and gradients of your model, corresponding to
    the inputs and targets provided. It should return the loss and gradients.
    """
    with tf.GradientTape() as tape:
        loss_value = loss(targets, model(inputs)[0])
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
