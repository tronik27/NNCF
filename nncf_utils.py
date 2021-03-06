from tensorflow.keras.layers import Input, Reshape, Layer, BatchNormalization, Conv2D, Dense, Flatten, Add, experimental
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from Correlation_utils import Plot3D


def make_gt_correlation(shape, num_corrs, labels, gt_lable):
    gt = np.zeros((len(labels), shape, shape, num_corrs), dtype='float32')
    if shape % 2:
        gt[np.where(labels == gt_lable), shape // 2 - 1:shape // 2 + 2, shape // 2 - 1:shape // 2 + 2, :] = 1
    else:
        gt[np.where(labels == gt_lable), shape//2-1:shape//2+1, shape//2-1:shape//2+1, :] = 1
    return gt


def make_dataset(images, labels, batch_size, augmentation=False):
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    data = data.shuffle(len(images), reshuffle_each_iteration=True)
    data = data.batch(batch_size, drop_remainder=True)
    if augmentation:
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_augmentation = tf.keras.Sequential([experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                                                 experimental.preprocessing.RandomRotation(0.2)])
        data = data.map(lambda x, y: (x, data_augmentation(y, training=True)), num_parallel_calls=AUTOTUNE)

    data = data.map(lambda x, y: (x, y[..., np.newaxis]))
    data = data.map(lambda x, y: (x, tf.transpose(y, perm=(0, 4, 1, 2, 3))))

    return data


def data_prepare(train_images, train_labels, validation_images=None, validation_labels=None, label=0, num_of_corr=32):

    if not validation_images.any():
        train_images, train_labels, validation_images, validation_labels = train_test_split(train_images,
                                                                                            train_labels,
                                                                                            shuffle=True,
                                                                                            test_size=0.33,
                                                                                            random_state=42)

    train_images = train_images[..., np.newaxis].astype(np.float32) / 255.
    train_images = train_images[train_labels == label]
    train_corr = make_gt_correlation(train_images.shape[1], num_of_corr, train_labels[train_labels == label], label)
    print(np.where(train_corr > 0))
    train_data = make_dataset(train_corr, train_images, num_of_corr, augmentation=False)

    validation_images = validation_images[..., np.newaxis].astype(np.float32) / 255.
    validation_images = validation_images[:2000, :, :, :]
    validation_num_of_corr = num_of_corr
    validation_corr = make_gt_correlation(validation_images.shape[1],
                                          validation_num_of_corr,
                                          validation_labels[:2000],
                                          label)
    validation_data = make_dataset(validation_corr, validation_images, num_of_corr, augmentation=True)

    return train_data, validation_data, train_images.shape


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