def get_preprocessing_parameters(fold_num):
    splits = ['sigtia-configuration2-splits/fold_1',
              'sigtia-configuration2-splits/fold_2',
              'sigtia-configuration2-splits/fold_3',
              'sigtia-configuration2-splits/fold_4',
              'sigtia-configuration2-splits/fold_benchmark',
              'single-note-splits']

    split = splits[fold_num]

    config = {'audio_path': '../MAPS',
              'train_fold': './splits/{}/train'.format(split),
              'valid_fold': './splits/{}/valid'.format(split),
              'test_fold': './splits/{}/test'.format(split),
              'tfrecords_train_fold': './tfrecords-dataset/{}/train/'.format(split),
              'tfrecords_valid_fold': './tfrecords-dataset/{}/valid/'.format(split),
              'tfrecords_test_fold': './tfrecords-dataset/{}/test/'.format(split),
              'chord_folder': './tfrecords-dataset/chords/',
              'chord_fold': './splits/UMA_chords/train',
              'chroma_folder': './chroma/',
              'context_frames': 2,
              'is_chroma': False,
              'is_hpcp': True,
              'audio_config': {'num_channels': 1,
                               'sample_rate': 44100,
                               'filterbank': 'LogarithmicFilterbank',
                               'frame_size': 4096,
                               'fft_size': 4096,
                               'fps': 100,
                               'num_bands': 48,
                               'fmin': 30.0,
                               'fmax': 4200.0,
                               'fref': 440.0,
                               'norm_filters': True,
                               'unique_filters': True,
                               'circular_shift': False,
                               'norm': True},
              'audio_config_2': {'num_channels': 1,
                                 'sample_rate': 44100,
                                 'filterbank': 'LogarithmicFilterbank',
                                 'frame_size': 4096,
                                 'fft_size': 4096,
                                 'fps': 100,
                                 'num_bands': 48,
                                 'fmin': 30,
                                 'fmax': 2000.0,
                                 'fref': 440.0,
                                 'norm_filters': True,
                                 'unique_filters': True,
                                 'circular_shift': False,
                                 'norm': True}
              }

    return config


def get_hpcp_parameters():
    config = {'num_channels': 1,
              'sample_rate': 44100,
              'frame_size': 4096,
              'fft_size': 4096,
              'fps': 100,
              'num_classes': 12,
              'fmin': [27.5, 55.0, 110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0], #[20.0, 53.0, 105.0, 213.0, 427.0, 855.0, 1710.0, 3420.0]
              'fmax': [51.9, 103.8, 207.7, 415.3, 830.6, 1661.2, 3322.4, 6644.9], #[53.0, 105.0, 213.0, 427.0, 855.0, 1710.0, 3420.0, 4300.0]
              'fref': 440.0,
              'window': 1,
              'norm_filters': False,
              'circular_shift': False,
              'norm': True}
    return config
