import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from nncf_utils import FilterDataset, make_tensorboard, SetWeightsCallback
from NNCF import NNCFModel


def get_callbacks(data):
    tensorboard_callback = make_tensorboard()
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=0.0001)
    set_weights = SetWeightsCallback(data)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('training_nncf/weights', save_weights_only=True,
                                                    save_best_only=True, monitor='val_loss', mode='min')
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)
    callbacks = [tensorboard_callback, reduce_lr, set_weights, checkpoint, stop]
    return callbacks


def get_metrics():
    mse = tf.keras.metrics.MeanAbsoluteError()
    cos_sim = tf.keras.metrics.CosineSimilarity(axis=1),
    lce = tf.keras.metrics.LogCoshError()
    metrics = [mse, cos_sim, lce]
    return metrics


num_correlations = 32
epochs = 35
num_of_images = 12000
initial_filter_matrix = tf.zeros((1, 28, 28, 1))
gt_label = 3

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
datagen = FilterDataset(gt_label=gt_label, num_of_corr=num_correlations,
                        num_of_images=num_of_images, augmentation=(0.2, True, True))
train_data, validation_data, shape = datagen.prepare_data_from_array(train_images=train_images,
                                                                     train_labels=train_labels,
                                                                     validation_images=test_images,
                                                                     validation_labels=test_labels)

nncf = NNCFModel(num_correlations=num_correlations,
                 initial_filter_matrix=initial_filter_matrix)
nncf.build(input_shape=shape)
print(nncf.summary())

nncf.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss=tf.keras.losses.MeanSquaredError(),
             metrics=get_metrics())

history = nncf.fit(train_data, validation_data=validation_data, steps_per_epoch=shape[0] // num_correlations,
                   callbacks=get_callbacks(train_data), epochs=epochs)

best_nncf = NNCFModel(num_correlations=num_correlations,
                      initial_filter_matrix=initial_filter_matrix)
best_nncf.build(input_shape=shape)
best_nncf.load_weights('training_nncf/weights')

cf = best_nncf.get_correlation_filter(plot=True)
best_nncf.plot_output_correlations()
