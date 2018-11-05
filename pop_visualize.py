import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import madmom
from numpy import loadtxt
from scipy import signal
import pypianoroll as ppr
from pypianoroll import Multitrack, Track
import glob
import librosa
import tensorflow as tf
import pop_input_data


def show_record(filepath):
    dataset = tf.data.TFRecordDataset(filepath)

    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()
    next_example = iterator.get_next()


    # Extract features from single example
    spec, labels = pop_input_data.tfrecord_train_parser(next_example)


    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharey=False)
    np_spec = np.zeros((231, 2, 3))
    np_label = np.zeros((88, 2))

    # Actrual session to run the graph.
    with tf.Session() as sess:
        for index in range(0, 400):
            try:
                spec_tensor, label_text = sess.run([spec, labels])

                example_slice = np.array(spec_tensor, np.float32)[:, 2, :].T
                #print(np.shape(example_slice))

                np_spec = np.append(np_spec, np.reshape(example_slice, (231, 1, 3)), axis=1)
                #rint(np.shape(np_spec))


                # Show the labels
                np_label = np.append(np_label, np.reshape(label_text, (88, 1)), axis=1)
                #print(label_text)


            except tf.errors.OutOfRangeError:
                break

        ax1.pcolormesh(np.flipud(np_spec[:, 150:, 0]))
        ax1.set_title("spec_512")

        ax2.pcolormesh(np.flipud(np_spec[:, 150:, 1]))
        ax2.set_title("spec_1024")

        ax3.pcolormesh(np.flipud(np_spec[:, 150:, 2]))
        ax3.set_title("spec_2048")

        ax4.pcolormesh(np_label[:, 150:])
        locs, l = plt.yticks()
        # plt.yticks(locs, np.arange(21, 108, 1))
        plt.grid(False)
        plt.show()


show_record(["/Users/Jaedicke/tensorflow/one_octave_resnet/training/29_train.tfrecords"])