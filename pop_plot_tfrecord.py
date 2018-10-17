import tensorflow as tf
import numpy as np
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

data_path = '208374_train.tfrecords'  # address to save the tfrecord file
with tf.Session() as sess:
    feature = {'train/spec': tf.FixedLenFeature([3960], tf.float32),
               'train/label': tf.FixedLenFeature([88], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    spec = tf.cast(features['train/spec'], tf.float32)

    # Cast label data into int32
    label = tf.cast(features['train/label'], tf.int32)
    # Reshape image data into the original shape
    spec = tf.reshape(spec, [88, 15, 3])

    # Any preprocessing here ...
    #spec = tf.image.per_image_standardization(spec)

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([spec, label], batch_size=10, capacity=30, num_threads=1,
                                            min_after_dequeue=10)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(5):
        img, lbl = sess.run([images, labels])
        #img = img.astype(np.uint8)
        for j in range(1):
            #plt.subplot(2, 3, j + 1)
            plt.imshow(img[j, ...]/np.max(img))
            print(lbl[j])
            #plt.title('cat' if lbl[j] == 0 else 'dog')
        plt.show()
    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()