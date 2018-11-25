from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from math import sqrt


def conv_net_model_fn(features, labels, mode, params):
    if params['data_format'] == 'NCHW':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        features = tf.reshape(features, [-1, params['num_channels'], params['frames'], params['freq_bins']])
    else:
        features = tf.reshape(features, [-1, params['frames'], params['freq_bins'], params['num_channels']])

    # learning_rate_fn = learning_rate_with_decay(
    #     initial_learning_rate=params['learning_rate'],
    #     batches_per_epoch=params['batches_per_epoch'],
    #     boundary_epochs=params['boundary_epochs'],
    #     decay_rates=params['decay_rates'])
    #
    # momentum_fn = momentum_with_decay(
    #     initial_momentum=params['momentum'],
    #     batches_per_epoch=params['batches_per_epoch'],
    #     boundary_epochs=params['boundary_epochs'],
    #     decay_rates=params['decay_rates_momentum'])

    learning_rate_fn = cycle_fn(params['learning_rate_cycle'], params['batches_per_epoch'], params['boundary_epochs'])

    momentum_fn = cycle_fn(params['momentum_cycle'], params['batches_per_epoch'], params['boundary_epochs'])

    # Empirical testing showed that including batch_normalization variables
    # in the calculation of regularized loss helped validation accuracy
    # for the CIFAR-10 dataset, perhaps because the regularization prevents
    # overfitting on the small data set. We therefore include all vars when
    # regularizing and computing loss during training.
    def loss_filter_fn(name):
        return 'conv' in name

    return conv_net_init(
        features=features,
        labels=labels,
        mode=mode,
        learning_rate_fn=learning_rate_fn,
        loss_filter_fn=loss_filter_fn,
        weight_decay=params['weight_decay'],
        momentum_fn=momentum_fn,
        clip_norm=params['clip_norm'],
        data_format=params['data_format'],
        batch_size=params['batch_size'],
        dtype=params['dtype']
    )


def cycle_fn(
        cycle_factor, batches_per_epoch, boundary_epochs):
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
    vals = [learning_rate for learning_rate in cycle_factor]

    def cycle_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return cycle_rate_fn


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


def momentum_with_decay(
        initial_momentum, batches_per_epoch, boundary_epochs, decay_rates):
    """Get a learning rate that decays step-wise as training progresses.

    Args:
      initial_momentum: The start momentum.
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
    vals = [initial_momentum * decay for decay in decay_rates]

    def momentum_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return momentum_fn


def weights_from_labels(labels):
    labeled_examples = np.where(labels == 1.0)
    weights = np.zeros(np.shape(labels))
    weights[labeled_examples[0], :] = 1.0
    weights[labeled_examples] = 2.0
    return np.where(weights == 0.0, 0.25, weights)


def conv_net_init(features, labels, mode, learning_rate_fn, loss_filter_fn, weight_decay, momentum_fn, clip_norm, data_format, batch_size, dtype=tf.float32):
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
    #tf.summary.image('images', features, max_outputs=6)

    features = tf.cast(features, dtype)

    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.cast(labels, dtype)

    logits = conv_net_kelz(features, mode == tf.estimator.ModeKeys.TRAIN, data_format=data_format, batch_size=batch_size)


    # Visualize conv1 kernels
    # with tf.variable_scope('conv1'):
    #     tf.get_variable_scope().reuse_variables()
    #     weights = tf.get_variable('weights')
    #     grid = put_kernels_on_grid(weights)
    #     tf.summary.image('conv1/kernels', grid, max_outputs=1)

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

    individual_loss = log_loss(labels, tf.clip_by_value(predictions['probabilities'], clip_norm, 1.0-clip_norm), epsilon=0.0)
    loss = tf.reduce_mean(individual_loss)

    # loss_filter_fn = loss_filter_fn
    #
    # # Add weight decay to the loss.
    # l2_loss = weight_decay * tf.add_n(
    #     # loss is computed using fp32 for numerical stability.
    #     [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
    #      if loss_filter_fn(v.name)])
    # l1_loss = tf.add_n(
    #     # loss is computed using fp32 for numerical stability.
    #     [l1_loss_fn(tf.cast(v, tf.float32), weight_decay, scope='l1_loss') for v in tf.trainable_variables()
    #      if loss_filter_fn(v.name)])
    # tf.summary.scalar('l2_loss', l2_loss)
    # tf.summary.scalar('l1_loss', l1_loss)
    # loss = loss + l1_loss + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)
        momentum = momentum_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        # Create a tensor named momentum for logging purposes
        tf.identity(momentum, name='momentum')
        tf.summary.scalar('momentum', momentum)

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


def conv_net_kelz(inputs, is_training, data_format='NHWC', batch_size=8):
    """Builds the ConvNet from Kelz 2016."""
    if data_format == 'NCHW':
        transpose_shape = [2, 1, 0]
    else:
        transpose_shape = [1, 0, 2]
    with slim.arg_scope(
          [slim.conv2d, slim.fully_connected],
          activation_fn=tf.nn.relu,
          weights_initializer=tf.contrib.layers.variance_scaling_initializer(
              factor=2.0, mode='FAN_AVG', uniform=True)):
        with slim.arg_scope([slim.batch_norm], is_training=is_training, data_format=data_format):
            net = slim.conv2d(
                inputs, 32, [3, 3], scope='conv1', normalizer_fn=slim.batch_norm, padding='SAME', data_format=data_format)
            #conv1_output = tf.unstack(net, num=batch_size, axis=0)
            #grid = put_kernels_on_grid(tf.expand_dims(tf.transpose(conv1_output[0], transpose_shape), 2))
            #tf.summary.image('conv1/output', grid, max_outputs=1)
            print(net.shape)

            net = slim.conv2d(
                net, 32, [3, 3], scope='conv2', normalizer_fn=slim.batch_norm, padding='VALID', data_format=data_format)
            #conv2_output = tf.unstack(net, num=batch_size, axis=0)
            #grid = put_kernels_on_grid(tf.expand_dims(tf.transpose(conv2_output[0], transpose_shape), 2))
            #tf.summary.image('conv2/output', grid, max_outputs=1)
            print(net.shape)
            net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2', data_format=data_format)
            print(net.shape)
            net = slim.dropout(net, 0.25, scope='dropout2', is_training=is_training)

            net = slim.conv2d(
                net, 64, [3, 3], scope='conv3', normalizer_fn=slim.batch_norm, padding='VALID', data_format=data_format)
            #conv3_output = tf.unstack(net, num=batch_size, axis=0)
            #grid = put_kernels_on_grid(tf.expand_dims(tf.transpose(conv3_output[0], transpose_shape), 2))
            #tf.summary.image('conv3/output', grid, max_outputs=1)
            print(net.shape)
            net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool3', data_format=data_format)

            net = slim.dropout(net, 0.25, scope='dropout3', is_training=is_training)

            # Flatten
            print(net.shape)
            net = tf.reshape(net, (-1, 64*1*55), 'flatten4')
            print(net.shape)
            net = slim.fully_connected(net, 512, scope='fc5')
            print(net.shape)
            net = slim.dropout(net, 0.5, scope='dropout5', is_training=is_training)
            net = slim.fully_connected(net, 88, activation_fn=None, scope='fc6')
            print(net.shape)
            return net


def log_loss(labels, predictions, epsilon=1e-7, scope=None, weights=None):
    """Calculate log losses.
        Same as tf.losses.log_loss except that this returns the individual losses
        instead of passing them into compute_weighted_loss and returning their
        weighted mean. This is useful for eval jobs that report the mean loss. By
        returning individual losses, that mean loss can be the same regardless of
        batch size.
        Args:
            labels: The ground truth output tensor, same dimensions as 'predictions'.
            predictions: The predicted outputs.
            epsilon: A small increment to add to avoid taking a log of zero.
            scope: The scope for the operations performed in computing the loss.
            weights: Weights to apply to labels.
        Returns:
            A `Tensor` representing the loss values.
        Raises:
            ValueError: If the shape of `predictions` doesn't match that of `labels`.
    """
    with tf.name_scope(scope, "log_loss", (predictions, labels)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = -tf.multiply(labels, tf.log(predictions + epsilon)) - tf.multiply(
            (1 - labels), tf.log(1 - predictions + epsilon))
        if weights is not None:
            losses = tf.multiply(losses, weights)

        return losses


def l1_loss_fn(tensor, weight=1.0, scope=None):
    """Define a L1Loss, useful for regularize, i.e. lasso.
    Args:
      tensor: tensor to regularize.
      weight: scale the loss by this factor.
      scope: Optional scope for name_scope.
    Returns:
      the L1 loss op.
    """
    with tf.name_scope(scope, 'L1Loss', [tensor]):
        weight = tf.convert_to_tensor(weight,
                                      dtype=tensor.dtype.base_dtype,
                                      name='loss_weight')
        loss = tf.multiply(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
        return loss


def put_kernels_on_grid(kernel, pad=1):

    """Visualize conv. layer filters or output as an image.
        Arranges filters into a grid, with some paddings between adjacent filters.
        Args:
            kernel: tensor of shape [Y, X, NumChannels, NumKernels]
            pad: number of black pixels around each filter (between them)
        Return:
            Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    """
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return i, int(n / i)
    (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
    #print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

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
