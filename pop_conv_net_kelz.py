from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def conv_net_model_fn(features, labels, mode, params):

    learning_rate_fn = learning_rate_with_decay(
        initial_learning_rate=params['learning_rate'], batches_per_epoch=params['batches_per_epoch'],
        boundary_epochs=[0.7, 2, 3],  # boundary_epochs=[100, 150, 200],
        decay_rates=[0.0006, 0.0006, 0.0006, 0.0006])#decay_rates=[1, 0.1, 0.01, 0.001])

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
    loss = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=labels)

    # weights masking to emphasize positive examples
    #cross_entropy_per_class = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=labels,
    #                                                          reduction=tf.losses.Reduction.NONE)
    #cross_entropy = tf.losses.compute_weighted_loss(cross_entropy_per_class, weights=weights)

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
        print(net.shape)

        net = slim.conv2d(
            net, 32, [3, 3], scope='conv2', normalizer_fn=slim.batch_norm, padding='VALID')
        print(net.shape)
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2')
        print(net.shape)
        net = slim.dropout(net, 0.25, scope='dropout2')

        net = slim.conv2d(
            net, 64, [3, 3], scope='conv3', normalizer_fn=slim.batch_norm, padding='VALID')
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
