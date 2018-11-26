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
              'context_frames': 2,
              'is_chroma': True,
              'audio_config': {'num_channels': 1,
                               'sample_rate': 44100,
                               'filterbank': 'LogarithmicFilterbank',
                               'frame_size': 4096,
                               'fft_size': 1024*4,
                               'fps': 100,
                               'num_bands': 48,
                               'fmin': 30,
                               'fmax': 8000.0,
                               'fref': 440.0,
                               'norm_filters': True,
                               'unique_filters': True,
                               'circular_shift': True,
                               'norm': True}
              }

    return config
