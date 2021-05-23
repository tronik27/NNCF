import numpy as np
import cv2 as cv
from numpy.fft import ifft2
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.filters import threshold_yen
from skimage.feature import hog
from tensorflow.keras.layers import Conv2D
import tensorflow as tf
from skimage.transform import resize
from typing import Tuple
from math import ceil


class Correlator:
    def __init__(self, image, cf, filter_plane='spatial', modulator_size=(1080, 1920)):
        self.cf = cf
        if len(image.shape) > 2:
            self.image = np.mean(image, axis=2)
        else:
            self.image = image
        self.filter_plane = filter_plane
        self.scene = np.zeros(2)
        self.modulator_height = modulator_size[0]
        self.modulator_width = modulator_size[1]

    def correlation(self):

        if self.cf.shape[0] > self.image.shape[0]:
            self.image = resize(self.image, self.cf.shape, anti_aliasing=True)

        if self.filter_plane == 'freq':
            corr_filter = np.expand_dims(np.abs(self._ifft2(self.cf)), axis=(2, 3))
        else:
            corr_filter = np.expand_dims(self.cf, axis=(2, 3))

        weights = np.transpose(np.expand_dims(self.image, axis=(2, 3, 4)), [4, 0, 1, 2, 3])
        cf_tensor = tf.convert_to_tensor(np.transpose(corr_filter, (3, 0, 1, 2)), dtype='float32')

        inputs = tf.keras.Input(shape=cf_tensor.shape[1:])
        correlation = Conv2D(filters=1, kernel_size=self.image.shape, use_bias=False,
                             padding='same', activation=None, trainable=False, name='correlation')(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=correlation)
        model.get_layer('correlation').set_weights(weights)
        self.scene = model(cf_tensor, training=False)[0, :, :, 0]

        return self.scene

    def fourier_correlation(self):

        if self.cf.shape[0] > self.image.shape[0]:
            self.image = resize(self.image, self.cf.shape, anti_aliasing=True)

        a, b = self.cf.shape
        c, d = self.image.shape
        l1 = a + c - 1
        l2 = b + d - 1
        img1 = np.zeros((l1, l2)).astype(complex)
        img2 = np.zeros((l1, l2))
        img1[(l1 - a) // 2:(l1 + a) // 2, (l2 - b) // 2:(l2 + b) // 2] = self.cf - np.mean(self.cf)
        img2[(l1 - c) // 2:(l1 + c) // 2, (l2 - d) // 2:(l2 + d) // 2] = self.image - np.mean(self.image)

        if self.filter_plane == 'freq':
            corr = np.abs(self._ifft2(self._fft2(img2) * np.conj(img1)))
        else:
            corr = np.abs(self._ifft2(self._fft2(img2) * np.conj(self._fft2(img1))))

        self.scene = np.copy(corr[(l1 - c) // 2:(l1 + c) // 2, (l2 - d) // 2:(l2 + d) // 2])
        return self.scene

    def van_der_lugt(self):

        cf, img = self._prepare_data_for_modulator(correlator_type='4f', sample_level=8)
        img1 = np.zeros(img.shape).astype(complex)
        img1[:cf.shape[0], :cf.shape[1]] = cf
        img2 = img

        self.scene = np.abs(self._ifft2(self._fft2(img2) * np.conj(self._fft2(img1))))
        return self.scene

    def joint_transform(self):

        cf = self._prepare_data_for_modulator(correlator_type='2f', sample_level=8)

        img = np.zeros((self.modulator_height, self.modulator_width)).astype(complex)
        img[:cf.shape[0], :cf.shape[1]] = cf
        img[-self.image.shape[0]:, -self.image.shape[1]:] = self.image

        self.scene = abs(self._ifft2(abs(self._fft2(img))))

        return self.scene

    @staticmethod
    def resize_image(image, max_height, max_wide):
        h, w = image.shape
        if (h > max_height) and (w > max_wide):
            if w > h:
                resized_image = cv.resize(image, (max_wide, int(h * float(max_wide) / w)),
                                          interpolation=cv.INTER_AREA)
            else:
                resized_image = cv.resize(image, (int(w * float(max_height) / h), max_height),
                                          interpolation=cv.INTER_AREA)
        elif h > max_height:
            resized_image = cv.resize(image, (int(w * float(max_height) / h), max_height),
                                      interpolation=cv.INTER_AREA)
        else:
            resized_image = cv.resize(image, (max_wide, int(h * float(max_wide) / w)), interpolation=cv.INTER_AREA)
        return resized_image

    def _prepare_data_for_modulator(self, correlator_type, sample_level=8):

        if self.cf.shape[0] >= self.image.shape[0]:
            self.image = resize(self.image, self.cf.shape, anti_aliasing=True)

        height = self.modulator_height // 2
        width = self.modulator_width // 2

        if (self.image.shape[0] > height) or (self.image.shape[1] > width):
            self.image = self._resize_image(self.image, height, width)

        if (self.cf.shape[0] > height) or (self.cf.shape[1] > width):
            self.cf = self._resize_image(self.cf, height, width)

        if correlator_type == '4f':
            cf = self._amplitude_hologram(sample_level=sample_level)
            img = np.zeros((2 * self.image.shape[0], 2 * self.image.shape[1]))
            img[:self.image.shape[0], :self.image.shape[1]] = self.image
        else:
            if self.filter_plane == 'freq':
                cf = np.abs(self._ifft2(self.cf))
                cf = (cf / np.max(cf) * (2 ** sample_level - 1)).astype(np.int)
            else:
                cf = (self.cf / np.max(self.cf) * (2 ** sample_level - 1)).astype(np.int)

        return cf, img

    def _amplitude_hologram(self, sample_level=8):
        H = np.zeros((2 * self.cf.shape[0], 2 * self.cf.shape[1])).astype(complex)
        if self.filter_plane == 'freq':
            H[:self.cf.shape[0], :self.cf.shape[1]] = self._ifft2(self.cf)
        else:
            H[:self.cf.shape[0], :self.cf.shape[1]] = self.cf
        H = self._fft2(H)
        holo = np.real(H) - np.min(np.real(H))
        holo = (holo / np.max(holo) * (2 ** sample_level - 1)).astype(np.int)
        return holo

    @staticmethod
    def _fft2(image: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
        """
        Direct 2D Fourier transform for the image.
        :param image: image array.
        :param axes: axes over which to compute the FFT.
        :return: Fourier transform of the image.
        """
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=axes), axes=axes), axes=axes)

    @staticmethod
    def _ifft2(image: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
        """
        Inverse 2D Fourier Transform for Image.
        :param image: image array.
        :param axes: axes over which to compute the FFT.
        :return: Fourier transform of the image.
        """
        return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image, axes=axes), axes=axes), axes=axes)

    def plot(self):
        if self.scene.any():
            plt.figure(figsize=(8, 3))
            ax1 = plt.subplot(1, 3, 1, adjustable='box')
            ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box')
            ax3 = plt.subplot(1, 3, 3)

            ax1.imshow(self.image, cmap='gray')
            ax1.set_axis_off()
            ax1.set_title('Scene with object')

            ax2.imshow(np.abs(self.cf), cmap='gray')
            ax2.set_axis_off()
            ax2.set_title('Correlation Filter')

            ax3.imshow(np.abs(self.scene))
            ax3.set_axis_off()
            ax3.set_title("Cross-correlation")
            plt.show()
        else:
            print('Nothing to show!')


class PlotCrossCorrelation:

    def __init__(self, corr_scenes, labels=np.zeros(3)):
        if len(corr_scenes.shape) < 3:
            self.corr_scenes = np.expand_dims(corr_scenes, axis=0)
        elif len(corr_scenes.shape) > 3:
            self.corr_scenes = np.mean(corr_scenes, axis=-1)
        else:
            self.corr_scenes = corr_scenes
        self.labels = labels

    def plot_3D(self):
        fig = plt.figure(figsize=(self.corr_scenes.shape[0]*5, 4))
        fig.suptitle('Cross-correlation', fontsize=16)
        for i in range(self.corr_scenes.shape[0]):
            axes = fig.add_subplot(1, self.corr_scenes.shape[0], i+1, projection='3d')
            x = np.arange(0, self.corr_scenes[i].shape[0], 1)
            y = np.arange(0, self.corr_scenes[i].shape[1], 1)
            x, y = np.meshgrid(x, y)
            surf = axes.plot_surface(x, y, self.corr_scenes[i], rstride=ceil(self.corr_scenes[i].shape[0] / 100),
                                     cstride=ceil(self.corr_scenes[i].shape[1] / 100), cmap=cm.jet)
            fig.colorbar(surf, shrink=0.5, aspect=10)
            if self.labels.any():
                if self.labels[i] == 1:
                    axes.set_title('Positive Correlation', size=15)
                else:
                    axes.set_title('Negative Correlation', size=15)
        plt.show()

    def plot(self):
        if self.corr_scenes.shape[0] > 4:
            if self.corr_scenes.shape[0] % 4:
                self.corr_scenes = self.corr_scenes[:-3, :, :]
            number_of_cols = self.corr_scenes.shape[0] // 4
            number_of_rows = 4
        elif self.corr_scenes.shape[0] <= 4:
            number_of_rows = 1
            number_of_cols = self.corr_scenes.shape[0]

        fig, axes = plt.subplots(nrows=number_of_rows, ncols=number_of_cols, figsize=(4 * number_of_rows,
                                                                                      4 * number_of_cols))
        axes = np.array(axes)
        for i, axe in enumerate(axes.flat):
            axe.imshow(self.corr_scenes[i, :, :], cmap=cm.jet)
            if self.labels.any():
                if self.labels[i] == 1:
                    axe.set_title('Positive Correlation', size=10)
                else:
                    axe.set_title('Negative Correlation', size=10)
        plt.show()
        
