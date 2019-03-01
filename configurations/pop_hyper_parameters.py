import tensorflow as tf


DEFAULT_DTYPE = tf.float32
# Context - 7
# fold_1: train - 4195653, valid - 748717, test - 1569405
# Context - 2
# fold_1: train - 4197453, valid - 749017, test - 1570005
# fold_2: train - 4249469, valid - 697001, test - 1570005
# fold_3: train - 4366098. valid - 580372, test - 1570005
# fold_4: train - 4221233. valid - 725237, test - 1570005

# Context - 10
# fold_1: train - 4194573, valid - 749017, test - 1569045

# RNN with 2000 frames
# fold_1: train - 2008, valid - , test - 752

# RNN with 2000 frames using also zero padded remainder
# fold_1: train - 2188, valid - , test - 812

num_examples = 2188
num_val_examples = 812
num_test_examples = 812
batch_size = 8 # 128 for conv, 8 for RNN
batches_per_epoch = int(round(num_examples/batch_size))
train_epochs = 150
total_train_steps = train_epochs * batches_per_epoch


def frange(start, stop, step):
    i = start
    if start < stop:
        while i < stop:
            yield i
            i += step
    else:
        while i > stop:
            yield i
            i -= step


def get_hyper_parameters(net):
    if net == 'ConvNet':
        config = {'use_rnn': False,  # default to false, there is no RNN with the ConvNet implementation
                  'use_architecture': 'convnet',
                  'batch_size': batch_size,
                  'dtype': DEFAULT_DTYPE,
                  'clip_norm': 1e-7,
                  # initial learning rate
                  'learning_rate': 1.0,
                  # when to change learning rate
                  'boundary_epochs': [5, 10, 15, 20, 25, 30, 35, 40], #[epoch for epoch in frange(0, train_epochs, train_epochs/60)][0:60],
                  # factor by which the initial learning rate is multiplied (needs to be one more than the boundaries)
                  'learning_rate_cycle': [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125, 0.000390625], #[learning_rate for learning_rate in frange(10e-5, 1., (1. - 10e-5) / (30. + 2.))][
                                         #0:30] + [learning_rate for learning_rate in frange(1., 10e-5, (1. - 10e-5) / (30. + 2.))][
                                         #0:31],
                  'decay_rates': [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 10e-1, 10e-2, 10e-3, 10e-4, 10e-3],
                  'momentum': 1.0,
                  'momentum_cycle': [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], #[momentum for momentum in frange(0.95, 0.85, (0.95-0.85)/(30+2))][0:30] + [momentum for momentum in frange(0.85, 0.95, (0.95-0.85)/(30+2))][0:31],
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
                  'weight_decay': 1e-7,
                  'data_format': 'NCHW', # NHWC (channels last, faster on CPU) or NCHW (channels first, faster on GPU)
                  'train_epochs': train_epochs}
    elif net == 'ResNet_v1':
        config = {'use_rnn': False,
                  'use_architecture': 'resnet',
                  'batch_size': batch_size,
                  'dtype': DEFAULT_DTYPE,
                  'clip_norm': 1e-7,
                  # initial learning rate
                  'learning_rate': 1.0,
                  # when to change learning rate
                  'boundary_epochs': [3, 6, 9, 12, 15, 18, 20, 22, 24],
                  # [epoch for epoch in frange(0, train_epochs, train_epochs/60)][0:60],
                  # factor by which the initial learning rate is multiplied (needs to be one more than the boundaries)
                  'learning_rate_cycle': [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125,
                                          0.000390625, 0.0001],
                  # [learning_rate for learning_rate in frange(10e-5, 1., (1. - 10e-5) / (30. + 2.))][
                  # 0:30] + [learning_rate for learning_rate in frange(1., 10e-5, (1. - 10e-5) / (30. + 2.))][
                  # 0:31],
                  'decay_rates': [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 10e-1, 10e-2, 10e-3, 10e-4, 10e-3],
                  'momentum': 1.0,
                  'momentum_cycle': [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                  # [momentum for momentum in frange(0.95, 0.85, (0.95-0.85)/(30+2))][0:30] + [momentum for momentum in frange(0.85, 0.95, (0.95-0.85)/(30+2))][0:31],
                  'frames': 15,
                  'freq_bins': 88, # 76 for octave-wise HPCP, 229 for log spec
                  'num_channels': 1,
                  'num_classes': 88,
                  'num_examples': num_examples,
                  'num_val_examples': num_val_examples,
                  'num_test_examples': num_test_examples,
                  'batches_per_epoch': batches_per_epoch,
                  'train_steps': total_train_steps,
                  'eval_steps': int(round(num_val_examples / batch_size)),
                  'test_steps': int(round(num_test_examples / batch_size)),
                  'weight_decay': 1e-7,
                  'data_format': 'channels_first', # this has to be 'channels_last' in case the RNN is used!  # NHWC (channels last, faster on CPU) or NCHW (channels first, faster on GPU)
                  'train_epochs': train_epochs}
    elif net == 'ResNet_v1_RNN':
        config = {'use_rnn': True,
                  'use_architecture': 'resnet',
                  'batch_size': 8, # change to 1 for inference, else 8
                  'dtype': DEFAULT_DTYPE,
                  'clip_norm': 1e-7,
                  # initial learning rate
                  'learning_rate': 1.0,
                  # when to change learning rate
                  'boundary_epochs': [3, 6, 9, 12, 15, 18, 20, 22, 24],
                  # [epoch for epoch in frange(0, train_epochs, train_epochs/60)][0:60],
                  # factor by which the initial learning rate is multiplied (needs to be one more than the boundaries)
                  'learning_rate_cycle': [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125,
                                          0.000390625, 0.0001],
                  # [learning_rate for learning_rate in frange(10e-5, 1., (1. - 10e-5) / (30. + 2.))][
                  # 0:30] + [learning_rate for learning_rate in frange(1., 10e-5, (1. - 10e-5) / (30. + 2.))][
                  # 0:31],
                  'decay_rates': [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 10e-1, 10e-2, 10e-3, 10e-4, 10e-3],
                  'momentum': 1.0,
                  'momentum_cycle': [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                  # [momentum for momentum in frange(0.95, 0.85, (0.95-0.85)/(30+2))][0:30] + [momentum for momentum in frange(0.85, 0.95, (0.95-0.85)/(30+2))][0:31],
                  'frames': 2000,
                  'freq_bins': 199, # 76 for octave-wise HPCP, 229 for log spec
                  'num_channels': 1,
                  'num_classes': 88,
                  'num_examples': num_examples,
                  'num_val_examples': num_val_examples,
                  'num_test_examples': num_test_examples,
                  'batches_per_epoch': batches_per_epoch,
                  'train_steps': total_train_steps,
                  'eval_steps': int(round(num_val_examples / batch_size)),
                  'test_steps': int(round(num_test_examples / batch_size)),
                  'weight_decay': 1e-7,
                  'data_format': 'channels_last', # this has to be 'channels_last' in case the RNN is used!  # NHWC (channels last, faster on CPU) or NCHW (channels first, faster on GPU)
                  'train_epochs': train_epochs}
    else:
        config = {}
    return config
