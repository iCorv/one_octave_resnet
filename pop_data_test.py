import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import madmom
import glob
import tensorflow as tf
import pop_input_data


def import_tfrecord(filepath):
    #dataset = tf.data.TFRecordDataset(filepath)
    input_shape = [5, 185, 1]
    num_labels = 88

    # Extract features from single example
    #spec, labels = pop_input_data.tfrecord_train_parser(next_example)
    #spec = tf.slice(spec, [0, 0, 1], [5, 231, 1])
    dataset = pop_input_data.tfrecord_test_input_fn(filepath, 1, 1)

    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()
    next_example = iterator.get_next()
    spec = next_example[0]
    labels = next_example[1]

    print(spec.shape)
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
    np_spec = np.zeros((input_shape[1], 2))
    np_label = np.zeros((num_labels, 2))

    # Actual session to run the graph.
    with tf.Session() as sess:
        for index in range(0, 1500):
            try:
                spec_tensor, label_text = sess.run([spec, labels])
                #print(spec_tensor.shape)
                example_slice = np.array(np.squeeze(spec_tensor[:,:,:,0]), np.float32)[1, :]
                #print(np.shape(example_slice))

                np_spec = np.append(np_spec, np.reshape(example_slice, (input_shape[1], 1)), axis=1)
                # print(np.shape(np_spec))

                # Show the labels
                np_label = np.append(np_label, np.reshape(label_text, (num_labels, 1)), axis=1)
                # print(label_text)


            except tf.errors.OutOfRangeError:
                break

        print(np.max(np_spec))
        print(np.min(np_spec))
        print(np.shape(np_spec))
        ax1.pcolormesh(np_spec[:, :])
        ax1.set_title("spec_4096")

        ax2.pcolormesh(np_label[:, :])
        locs, l = plt.yticks()
        # plt.yticks(locs, np.arange(21, 108, 1))
        plt.grid(False)
        plt.show()


def import_single_example(filepath):
    #dataset = tf.data.TFRecordDataset(filepath)

    input_shape = [5, 229]
    # Extract features from single example
    #spec, labels = pop_input_data.tfrecord_train_parser(next_example)
    #spec = tf.slice(spec, [0, 0, 1], [5, 231, 1])
    dataset = pop_input_data.tfrecord_test_input_fn(filepath, 8, 1)

    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()
    next_example = iterator.get_next()
    spec = next_example[0]
    labels = next_example[1]

    spec = tf.reshape(spec, [-1, input_shape[0], input_shape[1], 1])
    print(spec.shape)
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
    np_spec = np.zeros((input_shape[1], 1))
    np_label_fix = np.zeros((88, 2))
    label_mat = np.zeros((88, 1))

    # Actual session to run the graph.
    with tf.Session() as sess:
        for index in range(0, 100):
            try:
                spec_tensor, label_text = sess.run([spec, labels])
                #print(spec_tensor.shape)

                example_slice = np.array(np.squeeze(spec_tensor[3,:,:,:]), np.float32)
                #print(np.shape(example_slice))

                #np_spec = example_slice.T #np.reshape(example_slice, (229, 5))
                np_spec = np.append(np_spec, example_slice.T, axis=1)
                # print(np.shape(np_spec))

                # Show the labels
                label_mat = np.append(label_mat, np.append(np.append(np_label_fix, np.reshape(label_text[3,:], (88, 1)), axis=1), np_label_fix, axis=1), axis=1)
                # print(label_text)


            except tf.errors.OutOfRangeError:
                break

        print(np.max(np_spec))
        print(np.min(np_spec))
        print(np.shape(np_spec))
        ax1.pcolormesh(np_spec[:, 1:])
        ax1.set_title("spec_4096")

        ax2.pcolormesh(label_mat[:, 1:])
        locs, l = plt.yticks()
        # plt.yticks(locs, np.arange(21, 108, 1))
        plt.grid(False)
        plt.show()


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

    # Actual session to run the graph.
    with tf.Session() as sess:
        for index in range(0, 5000):
            try:
                spec_tensor, label_text = sess.run([spec, labels])
                #print(spec_tensor.shape)
                example_slice = np.array(spec_tensor, np.float32)[:, 2, :]
                #print(np.shape(example_slice))

                np_spec = np.append(np_spec, np.reshape(example_slice, (231, 1, 3)), axis=1)
                print(np.shape(np_spec))


                # Show the labels
                np_label = np.append(np_label, np.reshape(label_text, (88, 1)), axis=1)
                #print(label_text)


            except tf.errors.OutOfRangeError:
                break

        print(np.max(np_spec))
        print(np.min(np_spec))
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


#show_record(["/Users/Jaedicke/tensorflow/one_octave_resnet/training/29_train.tfrecords"])
#show_record(["D:/Users/cjaedicke/one_octave_resnet/maps_mus_train/100_train.tfrecords"])

import_tfrecord(["./tfrecords-dataset/sigtia-configuration2-splits/fold_benchmark/valid/MAPS_MUS-chpn_op66_AkPnBcht.tfrecords"])