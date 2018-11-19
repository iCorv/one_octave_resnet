from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from official.resnet import resnet_model
import pop_resnet_model_fn
import tensorflow as tf


class ResNet(resnet_model.Model):
    """Model class

    resnet_size should be either 18, 34, 50, 101, 152, 200
    """

    def __init__(self, resnet_size, num_classes,
                 resnet_version=resnet_model.DEFAULT_VERSION,
                 dtype=resnet_model.DEFAULT_DTYPE, data_format=None):
        """

        Args:
          resnet_size: The number of convolutional layers needed in the model.
          data_format: Either 'channels_first' or 'channels_last', specifying which
            data format to use when setting up the model.
          num_classes: The number of output classes needed from the model. This
            enables users to extend the same model to their own datasets.
          resnet_version: Integer representing which version of the ResNet network
          to use. Valid values: [1, 2]
          dtype: The TensorFlow dtype to use for calculations.
        """
        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            bottleneck = False
        else:
            bottleneck = True

        super(ResNet, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            block_sizes=_get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            resnet_version=resnet_version,
            data_format=data_format,
            dtype=dtype
        )


def _get_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.
    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
    Args:
      resnet_size: The number of convolutional layers needed in the model.
    Returns:
      A list of block sizes to use in building the model.
    Raises:
      KeyError: if invalid resnet_size is received.
    """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
               'Size received: {}; sizes allowed: {}.'.format(
            resnet_size, choices.keys()))
        raise ValueError(err)


def resnet_model_fn(features, labels, mode, params):
    if params['data_format'] == 'channels_first':
        features = tf.reshape(features, [-1, params['num_channels'], params['frames'], params['freq_bins']])
    else:
        features = tf.reshape(features, [-1, params['frames'], params['freq_bins'], params['num_channels']])

    learning_rate_fn = pop_resnet_model_fn.learning_rate_with_decay(
        initial_learning_rate=params['learning_rate'],
        batches_per_epoch=params['batches_per_epoch'],
        boundary_epochs=params['boundary_epochs'],
        decay_rates=params['decay_rates'])

    # Empirical testing showed that including batch_normalization variables
    # in the calculation of regularized loss helped validation accuracy
    # for the CIFAR-10 dataset, perhaps because the regularization prevents
    # overfitting on the small data set. We therefore include all vars when
    # regularizing and computing loss during training.
    def loss_filter_fn(_):
        return True

    return pop_resnet_model_fn.resnet_model_fn(
        features=features,
        labels=labels,
        mode=mode,
        model_class=ResNet,
        resnet_size=params['resnet_size'],
        weight_decay=params['weight_decay'],
        learning_rate_fn=learning_rate_fn,
        momentum=params['momentum'],
        data_format=params['data_format'],
        resnet_version=params['resnet_version'],
        loss_scale=params['loss_scale'],
        loss_filter_fn=loss_filter_fn,
        dtype=params['dtype'],
        num_classes=params['num_classes']
    )
