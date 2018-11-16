from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from math import sqrt


def conv_net_model_fn(features, labels, mode, params):
    features = tf.reshape(features, [-1, params['frames'], params['freq_bins'], params['num_channels']])
    learning_rate_fn = learning_rate_with_decay(
        initial_learning_rate=params['learning_rate'],
        batches_per_epoch=params['batches_per_epoch'],
        boundary_epochs=params['boundary_epochs'],
        decay_rates=params['decay_rates'])

    return conv_net_init(
        features=features,
        labels=labels,
        mode=mode,
        learning_rate_fn=learning_rate_fn,
        momentum=params['momentum'],
        clip_norm=params['clip_norm'],
        dtype=params['dtype']
    )


def learning_rate_with_decay(
        initial_learning_rate, batches_per_epoch, boundary_epochs, decay_rates):
    """Get a learning rate that decays step-wise as training progresses.

    Args:
      initial_learning_rate: The start learning rate.
      batches_per_epoch: number of batches per epoch, sometimes called steps
      per epoch.
      boundary_epochs: list of ints representing the epochs at which we
        decay the learning rate.
      decay_rates: list of floats representing the decay rates to be used
        for scaling the learning rate. It should have one more element
        than `boundary_epochs`, and all elements should have the same type.

    Returns:
      Returns a function that takes a single argument - the number of batches
      trained so far (global_step)- and returns the learning rate to be used
      for training the next batch.
    """

    # Reduce the learning rate at certain epochs, for Example:
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn


def weights_from_labels(labels):
    labeled_examples = np.where(labels == 1.0)
    weights = np.zeros(np.shape(labels))
    weights[labeled_examples[0], :] = 1.0
    weights[labeled_examples] = 2.0
    return np.where(weights == 0.0, 0.25, weights)


def conv_net_init(features, labels, mode, learning_rate_fn, momentum, clip_norm, dtype=tf.float32):
    """Shared functionality for different model_fns.

    Initializes the ConvNet representing the model layers
    and uses that model to build the necessary EstimatorSpecs for
    the `mode` in question. For training, this means building losses,
    the optimizer, and the train op that get passed into the EstimatorSpec.
    For evaluation and prediction, the EstimatorSpec is returned without
    a train op, but with the necessary parameters for the given mode.

    Args:
      features: tensor representing input images
      labels: tensor representing class labels for all input images
      mode: current estimator mode; should be one of
        `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
      learning_rate_fn: returns the learning rate to be used
      for training the next batch.
      momentum: the momentum imposed by the optimizer.
      clip_norm: the value the logistic function of the output layer is clipped by.
      dtype: the TensorFlow dtype to use for calculations.

    Returns:
      EstimatorSpec parameterized according to the input params and the
      current mode.
    """

    # Generate a summary node for the images
    tf.summary.image('images', features, max_outputs=6)

    features = tf.cast(features, dtype)

    if mode != tf.estimator.ModeKeys.PREDICT:
        # determine weights from labels encoding weights
        #weights = tf.py_func(weights_from_labels, [labels], [tf.float64], stateful=False)[0]
        #weights = tf.cast(weights, dtype=dtype)
        # since labels also encode the weights, we have to transform them to a binary format for evaluation
        #labels = tf.ceil(labels)
        labels = tf.cast(labels, dtype)

    #logits = cnn_model(features, mode == tf.estimator.ModeKeys.TRAIN)
    logits = conv_net_kelz(features)

    # Visualize conv1 kernels
    with tf.variable_scope('conv1'):
        tf.get_variable_scope().reuse_variables()
        weights = tf.get_variable('weights')
        grid = put_kernels_on_grid(weights)
        tf.summary.image('conv1/kernels', grid, max_outputs=1)

    # # Visualize conv1 kernels
    # with tf.variable_scope('conv2'):
    #     tf.get_variable_scope().reuse_variables()
    #     weights = tf.get_variable('weights')
    #     grid = put_kernels_on_grid(weights)
    #     tf.summary.image('conv2/kernels', grid, max_outputs=1)

    # This acts as a no-op if the logits are already in fp32 (provided logits are
    # not a SparseTensor). If dtype is of low precision, logits must be cast to
    # fp32 for numerical stability.
    logits = tf.cast(logits, tf.float32)

    predictions = {
        'classes': tf.round(tf.sigmoid(logits)),
        'probabilities': tf.sigmoid(logits, name='sigmoid_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    logits = tf.clip_by_value(logits, clip_norm, 1.0 - clip_norm)
    # without weights
    #loss = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=labels)

    # weights masking to emphasize positive examples
    cross_entropy_per_class = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=labels,
                                                              reduction=tf.losses.Reduction.NONE)
    loss = tf.losses.compute_weighted_loss(cross_entropy_per_class, weights=tf.add(tf.multiply(labels, 1.0), 0.2))

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        #optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum,
            use_nesterov=True
        )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            minimize_op = optimizer.minimize(loss, global_step)

        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    fn = tf.metrics.false_negatives(labels, predictions['classes'])
    fp = tf.metrics.false_positives(labels, predictions['classes'])
    tp = tf.metrics.true_positives(labels, predictions['classes'])
    precision = tf.metrics.precision(labels, predictions['classes'])
    recall = tf.metrics.recall(labels, predictions['classes'])
    # this is the Kelz et. al. def of frame wise metric F1
    f = tf.multiply(tf.constant(2.0), tf.multiply(precision[0], recall[0]))
    f = tf.divide(f, tf.add(precision[0], recall[0]))

    tf.identity(fn[0], name="fn")
    tf.identity(fp[0], name="fp")
    tf.identity(tp[0], name="tp")
    tf.identity(f, name="f1_score")

    # collect metrics
    metrics = {'false_negatives': fn,
               'false_positives': fp,
               'true_positives': tp,
               'precision': precision,
               'recall': recall,
               'f1_score': (f, tf.group(precision[1], recall[1]))}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def conv_net_kelz(inputs):
    """Builds the ConvNet from Kelz 2016."""
    with slim.arg_scope(
          [slim.conv2d, slim.fully_connected],
          activation_fn=tf.nn.relu,
          weights_initializer=tf.contrib.layers.variance_scaling_initializer(
              factor=2.0, mode='FAN_AVG', uniform=True)):
        net = slim.conv2d(
            inputs, 32, [3, 3], scope='conv1', normalizer_fn=slim.batch_norm, padding='SAME')
        conv1_output = tf.unstack(net, num=8, axis=0)
        grid = put_kernels_on_grid(tf.expand_dims(conv1_output[0], 2))
        tf.summary.image('conv1/output', grid, max_outputs=1)
        print(net.shape)

        net = slim.conv2d(
            net, 32, [3, 3], scope='conv2', normalizer_fn=slim.batch_norm, padding='VALID')
        conv2_output = tf.unstack(net, num=8, axis=0)
        grid = put_kernels_on_grid(tf.expand_dims(tf.transpose(conv2_output[0], [1, 0, 2]), 2))
        tf.summary.image('conv2/output', grid, max_outputs=1)
        print(net.shape)
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2')
        print(net.shape)
        net = slim.dropout(net, 0.25, scope='dropout2')

        net = slim.conv2d(
            net, 64, [3, 3], scope='conv3', normalizer_fn=slim.batch_norm, padding='VALID')
        conv3_output = tf.unstack(net, num=8, axis=0)
        grid = put_kernels_on_grid(tf.expand_dims(conv3_output[0], 2))
        tf.summary.image('conv3/output', grid, max_outputs=1)
        print(net.shape)
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool3')

        net = slim.dropout(net, 0.25, scope='dropout3')

        # Flatten
        print(net.shape)
        net = tf.reshape(net, (-1, 64*1*55), 'flatten4')
        print(net.shape)
        net = slim.fully_connected(net, 512, scope='fc5')
        print(net.shape)
        net = slim.dropout(net, 0.5, scope='dropout5')
        net = slim.fully_connected(net, 88, activation_fn=None, scope='fc6')
        print(net.shape)
        return net


def put_kernels_on_grid (kernel, pad = 1):

    '''Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.
    Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
    Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
    print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x
