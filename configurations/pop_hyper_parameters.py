import tensorflow as tf
import numpy as np


DEFAULT_DTYPE = tf.float32

num_examples = 4197453  # 4197453  # 482952
num_val_examples = 749017  # 749017  # 87628
num_test_examples = 1570005
batch_size = 128
batches_per_epoch = int(round(num_examples/batch_size))
train_epochs = 1
total_train_steps = train_epochs * batches_per_epoch


def get_hyper_parameters(net):
    if net == 'ConvNet':
        config = {'batch_size': batch_size,
                  'dtype': DEFAULT_DTYPE,
                  'clip_norm': 1e-7,
                  # initial learning rate
                  'learning_rate': 0.1,
                  # when to change learning rate
                  'boundary_epochs': [10, 20, 30],
                  # factor by which the initial learning rate is multiplied (needs to be one more than the boundaries)
                  'decay_rates': [1., 0.5, 0.5*0.5, 0.5*0.5*0.5],
                  'momentum': 0.9,
                  'frames': 5,
                  'freq_bins': 229,
                  'num_channels': 1,
                  'num_classes': 88,
                  'num_examples': num_examples,
                  'num_val_examples': num_val_examples,
                  'num_test_examples': num_test_examples,
                  'batches_per_epoch': batches_per_epoch,
                  'train_steps': total_train_steps,
                  'eval_steps': int(round(num_val_examples/batch_size)),
                  'test_steps': int(round(num_test_examples/batch_size)),
                  'data_format': 'NCHW', # NHWC (channels last, faster on CPU) or NCHW (channels first, faster on GPU)
                  'train_epochs': train_epochs}
    elif net == 'ResNet':
        config = {'batch_size': batch_size,
                  'dtype': DEFAULT_DTYPE,
                  'clip_norm': 1e-7,
                  # initial learning rate
                  'learning_rate': 1.0,
                  # when to change learning rate
                  'boundary_epochs': [3, 10, 20, 30],
                  # factor by which the initial learning rate is multiplied (needs to be one more than the boundaries)
                  'decay_rates':  [1., 0.1, 0.1 * 0.5, 0.1 * 0.5 * 0.5, 0.1 * 0.5 * 0.5 * 0.5],
                  'momentum': 0.9,
                  'frames': 5,
                  'freq_bins': 229,
                  'num_channels': 1,
                  'num_classes': 88,
                  'num_examples': num_examples,
                  'num_val_examples': num_val_examples,
                  'num_test_examples': num_test_examples,
                  'batches_per_epoch': batches_per_epoch,
                  'train_steps': total_train_steps,
                  'eval_steps': int(round(num_val_examples / batch_size)),
                  'test_steps': int(round(num_test_examples / batch_size)),
                  # We use a weight decay of 0.0002, which performs better
                  # than the 0.0001 that was originally suggested.
                  'weight_decay': 2e-4,
                  'resnet_size': 18,
                  'resnet_version': 2,
                  'loss_scale': 128 if DEFAULT_DTYPE == tf.float16 else 1,
                  # channels_last (channels last, faster on CPU) or channels_first (channels first, faster on GPU)
                  'data_format': 'channels_first',
                  'train_epochs': train_epochs}
    elif net == 'ResNet_range_test':
        config = {'batch_size': batch_size,
                  'dtype': DEFAULT_DTYPE,
                  'clip_norm': 1e-7,
                  # initial learning rate
                  'learning_rate': 1.0,
                  # when to change learning rate
                  'boundary_epochs': np.arange(0, 1, 1/batches_per_epoch),
                  # factor by which the initial learning rate is multiplied (needs to be one more than the boundaries)
                  'decay_rates': np.arange(10e-5, 10, (10-10e-5)/(batches_per_epoch+1)),
                  'momentum': 0.9,
                  'frames': 5,
                  'freq_bins': 229,
                  'num_channels': 1,
                  'num_classes': 88,
                  'num_examples': num_examples,
                  'num_val_examples': num_val_examples,
                  'num_test_examples': num_test_examples,
                  'batches_per_epoch': batches_per_epoch,
                  'train_steps': total_train_steps,
                  'eval_steps': int(round(num_val_examples / batch_size)),
                  'test_steps': int(round(num_test_examples / batch_size)),
                  # We use a weight decay of 0.0002, which performs better
                  # than the 0.0001 that was originally suggested.
                  'weight_decay': 2e-4,
                  'resnet_size': 18,
                  'resnet_version': 2,
                  'loss_scale': 128 if DEFAULT_DTYPE == tf.float16 else 1,
                  # channels_last (channels last, faster on CPU) or channels_first (channels first, faster on GPU)
                  'data_format': 'channels_first',
                  'train_epochs': train_epochs}
    else:
        config = {}
    return config
