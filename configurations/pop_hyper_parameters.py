import tensorflow as tf


DEFAULT_DTYPE = tf.float32

num_examples = 482952  # 4197453  # 482952
num_val_examples = 87628  # 749017  # 87628
batch_size = 8
batches_per_epoch = int(round(num_examples/batch_size))
train_epochs = 40
total_train_steps = train_epochs * batches_per_epoch


def get_hyper_parameters(net):
    if net == 'ConvNet':
        config = {'batch_size': batch_size,
                  'dtype': DEFAULT_DTYPE,
                  'clip_norm': 1e-7,
                  # initial learning rate
                  'learning_rate': 0.1,
                  # when to change learning rate
                  'boundary_epochs': [10, 20, 70],
                  # factor by which the initial learning rate is multiplied (needs to be one more than the boundaries)
                  'decay_rates': [1., 0.5, 0.5*0.5, 0.5*0.5*0.5],
                  'momentum': 0.9,
                  'frames': 5,
                  'freq_bins': 229,
                  'num_channels': 1,
                  'num_classes': 88,
                  'num_examples': num_examples,
                  'num_val_examples': num_val_examples,
                  'batches_per_epoch': batches_per_epoch,
                  'train_steps': total_train_steps,
                  'eval_steps': int(round(num_val_examples/batch_size)),
                  'data_format': 'channels_last',
                  'train_epochs': train_epochs}
    elif net == 'ResNet':
        config = {}
    else:
        config = {}
    return config