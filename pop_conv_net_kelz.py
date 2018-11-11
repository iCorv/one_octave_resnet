from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

matplotlib.use('TkAgg')
_HEIGHT = 5
_WIDTH = 229
_NUM_CHANNELS = 1
_NUM_CLASSES = 88
_NUM_IMAGES = {
    'train': 1042876, #4163882,
    'validation': 71435, #792567,
}


def conv_net_model_fn(features, labels, mode, params):
    features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

    learning_rate_fn = learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=params['batch_size'],
        num_images=_NUM_IMAGES['train'], boundary_epochs=[5, 5, 5],  # boundary_epochs=[100, 150, 200],
        decay_rates=[1, 0.1, 0.01, 0.001])

    # We use a weight decay of 0.0002, which performs better
    # than the 0.0001 that was originally suggested.
    weight_decay = 2e-4

    # Empirical testing showed that including batch_normalization variables
    # in the calculation of regularized loss helped validation accuracy
    # for the CIFAR-10 dataset, perhaps because the regularization prevents
    # overfitting on the small data set. We therefore include all vars when
    # regularizing and computing loss during training.
    def loss_filter_fn(_):
        return True

    return conv_net_prep(
        features=features,
        labels=labels,
        mode=mode,
        resnet_size=params['resnet_size'],
        weight_decay=weight_decay,
        learning_rate_fn=learning_rate_fn,
        momentum=0.9,
        data_format=params['data_format'],
        resnet_version=params['resnet_version'],
        loss_scale=params['loss_scale'],
        loss_filter_fn=loss_filter_fn,
        dtype=params['dtype'],
        num_classes=params['num_classes']
    )


def learning_rate_with_decay(
        batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
    """Get a learning rate that decays step-wise as training progresses.

    Args:
      batch_size: the number of examples processed in each training batch.
      batch_denom: this value will be used to scale the base learning rate.
        `0.1 * batch size` is divided by this number, such that when
        batch_denom == batch_size, the initial learning rate will be 0.1.
      num_images: total number of images that will be used for training.
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
    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Reduce the learning rate at certain epochs, for Example:
    # CIFAR-10: divide by 10 at epoch 100, 150, and 200
    # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
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


def conv_net_prep(features, labels, mode,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, resnet_version, loss_scale, num_classes,
                    loss_filter_fn=None, dtype=tf.float32):
    """Shared functionality for different model_fns.

    Initializes the ResnetModel representing the model layers
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
      model_class: a class representing a TensorFlow model that has a __call__
        function. We assume here that this is a subclass of ResnetModel.
      resnet_size: A single integer for the size of the ResNet model.
      weight_decay: weight decay loss rate used to regularize learned variables.
      learning_rate_fn: function that returns the current learning rate given
        the current global_step
      momentum: momentum term used for optimization
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      resnet_version: Integer representing which version of the ResNet network to
        use. Valid values: [1, 2]
      loss_scale: The factor to scale the loss for numerical stability.
      loss_filter_fn: function that takes a string variable name and returns
        True if the var should be included in loss calculation, and False
        otherwise. If None, batch_normalization variables will be excluded
        from the loss.
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

    logits = cnn_model(features, 0.99, mode == tf.estimator.ModeKeys.TRAIN)
    #logits = conv_net_kelz(features)

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
            #export_outputs={
            #    'predict': tf.estimator.export.PredictOutput(predictions)
            #})

    _EPSILON = tf.constant(1e-7, dtype=tf.float32)
    logits = tf.clip_by_value(logits, _EPSILON, 1.0 - _EPSILON)
    # without weights
    cross_entropy = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=labels)

    # weights masking to emphasize positive examples
    #cross_entropy_per_class = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=labels,
    #                                                          reduction=tf.losses.Reduction.NONE)
    #cross_entropy = tf.losses.compute_weighted_loss(cross_entropy_per_class, weights=weights)



    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    loss_filter_fn = loss_filter_fn or exclude_batch_norm

    # Add weight decay to the loss.
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
         if loss_filter_fn(v.name)])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss
    #loss = cross_entropy

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

        if loss_scale != 1:
            # When computing fp16 gradients, often intermediate tensor values are
            # so small, they underflow to 0. To avoid this, we multiply the loss by
            # loss_scale to make these tensor values loss_scale times bigger.
            scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

            # Once the gradient computation is complete we can scale the gradients
            # back to the correct scale before passing them to the optimizer.
            unscaled_grad_vars = [(grad / loss_scale, var)
                                  for grad, var in scaled_grad_vars]
            with tf.control_dependencies(update_ops):
                minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
        else:
            with tf.control_dependencies(update_ops):
                minimize_op = optimizer.minimize(loss, global_step)

        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    # metrics for evaluation
    # correct_prediction = tf.equal(predictions['classes'], labels)
    # accuracy2 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # a = tf.metrics.accuracy(labels, predictions['classes'])
    # tf.summary.scalar('accuracy2', accuracy2)
    # fn = tf.metrics.false_negatives(labels, predictions['classes'])
    # fp = tf.metrics.false_positives(labels, predictions['classes'])
    # tn = tf.metrics.true_negatives(labels, predictions['classes'])
    # f = tf.metrics.true_positives(labels, predictions['classes'])
    # p = tf.metrics.precision(labels, predictions['classes'])
    # r = tf.metrics.recall(labels, predictions['classes'])

    fn = tf.metrics.false_negatives(labels, predictions['classes'])
    fp = tf.metrics.false_positives(labels, predictions['classes'])
    tp = tf.metrics.true_positives(labels, predictions['classes'])
    precision = tf.metrics.precision(labels, predictions['classes'])
    recall = tf.metrics.recall(labels, predictions['classes'])
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


def cnn_model(input_layer, momentum, is_training):
    """Builds the ConvNet from Kelz 2016."""

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(input_layer, 32, (3, 3), activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), strides=(1, 1), padding='same', name='conv1')
    conv1 = tf.layers.batch_normalization(conv1, momentum=momentum, training=is_training, name='conv1_bn')
    print(conv1.shape)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(conv1, 32, (3, 3), activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), strides=(1, 1), padding='valid', name='conv2')
    conv2 = tf.layers.batch_normalization(conv2, momentum=momentum, training=is_training, name='conv2_bn')
    print(conv2.shape)

    # save image
    tf.summary.image("conv2", tf.slice(conv2, [0, 0, 0, 0], [-1, _HEIGHT, _WIDTH, 1]), 1, collections=['train'])

    # Pooling layer #1 - down-sample by 2X over freq.
    conv2_pool = tf.layers.max_pooling2d(conv2, (1, 2), strides=(1, 2), padding='valid', name='pool1')
    print(conv2_pool.shape)

    # dropout layer
    conv2_pool = tf.layers.dropout(conv2_pool, rate=0.25, noise_shape=None, seed=None, training=is_training,
                                   name='dropout1')

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(conv2_pool, 64, (3, 3), activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), strides=(1, 1), padding='valid', name='conv3')
    conv3 = tf.layers.batch_normalization(conv3, momentum=momentum, training=is_training, name='conv3_bn')
    print(conv3.shape)

    # Pooling layer #2 - down-sample by 2X over freq.
    conv3_pool = tf.layers.max_pooling2d(conv3, (1, 2), strides=(1, 2), padding='valid', name='pool2')
    print(conv3_pool.shape)

    # dropout layer
    conv3_pool = tf.layers.dropout(conv3_pool, rate=0.25, noise_shape=None, seed=None, training=is_training,
                                   name='dropout2')

    # dense layer
    conv3_pool = tf.reshape(conv3_pool, [-1, 64*1*55])
    print(conv3_pool.shape)
    dense = tf.layers.dense(conv3_pool,  activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), units=512, name='dense')
    dense = tf.layers.batch_normalization(dense, momentum=momentum, training=is_training, name='dense_bn')

    dense = tf.layers.dropout(dense, rate=0.5, noise_shape=None, seed=None, training=is_training,
                              name='dropout3')
    print(dense.shape)

    # logits layer
    logits = tf.layers.dense(dense, units=88, name='logits')

    return logits


def conv_net_kelz(inputs):
  """Builds the ConvNet from Kelz 2016."""
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu,
      weights_initializer=tf.contrib.layers.variance_scaling_initializer(
          factor=2.0, mode='FAN_AVG', uniform=True)):
    net = slim.conv2d(
        inputs, 32, [3, 3], scope='conv1', normalizer_fn=slim.batch_norm)
    print(net.shape)

    net = slim.conv2d(
        net, 32, [3, 3], scope='conv2', normalizer_fn=slim.batch_norm)
    print(net.shape)
    net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2')
    print(net.shape)
    net = slim.dropout(net, 0.25, scope='dropout2')

    net = slim.conv2d(
        net, 64, [3, 3], scope='conv3', normalizer_fn=slim.batch_norm)
    print(net.shape)
    net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool3')

    net = slim.dropout(net, 0.25, scope='dropout3')

    # Flatten while preserving batch and time dimensions.
    dims = tf.shape(net)
    print(net.shape)
    net = tf.reshape(net, (dims[0], 5,
                           net.shape[2].value * net.shape[3].value), 'flatten4')
    print(net.shape)
    net = slim.fully_connected(net, 512, scope='fc5')
    print(net.shape)
    net = slim.dropout(net, 0.5, scope='dropout5')
    net = tf.reshape(net, (dims[0], 5*512), 'flatten4')
    net = slim.fully_connected(net, 88, scope='fc6')
    print(net.shape)
    return net


def metrics_fn(predictions, labels, name=None):
    p, p_op = tf.metrics.precision(labels, predictions)
    r, r_op = tf.metrics.recall(labels, predictions)
    f = tf.multiply(tf.constant(2.0), tf.multiply(p, r))
    f = tf.divide(f, tf.add(p,r))
    #p = tf.identity(p, name="precision")

    return (p, p_op), (r, r_op), (f, tf.group(p_op,r_op))