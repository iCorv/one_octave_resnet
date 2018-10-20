from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from official.resnet import resnet_model
import pop_resnet_model_fn
import tensorflow as tf


_HEIGHT = 88
_WIDTH = 15
_NUM_CHANNELS = 3
_NUM_CLASSES = 88
_NUM_IMAGES = {
    'train': 208374,
    'validation': 38678,
}


class ResNet(resnet_model.Model):
    """Model class

    resnet_size should be either 18, 34, 50, 101, 152, 200
    """

    def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
                 resnet_version=resnet_model.DEFAULT_VERSION,
                 dtype=resnet_model.DEFAULT_DTYPE):
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
            first_pool_size=2,
            first_pool_stride=1,
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
    features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

    learning_rate_fn = pop_resnet_model_fn.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=params['batch_size'],
        num_images=_NUM_IMAGES['train'], boundary_epochs=[2, 3, 4],  # boundary_epochs=[100, 150, 200],
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

    return pop_resnet_model_fn.resnet_model_fn(
        features=features,
        labels=labels,
        mode=mode,
        model_class=ResNet,
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
