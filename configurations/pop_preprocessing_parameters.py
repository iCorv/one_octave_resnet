def get_preprocessing_parameters(fold_num):
    splits = ['sigtia-configuration2-splits/fold_1',
              'sigtia-configuration2-splits/fold_2',
              'sigtia-configuration2-splits/fold_3',
              'sigtia-configuration2-splits/fold_4',
              'sigtia-configuration2-splits/fold_benchmark',
              'single-note-splits']

    split = splits[fold_num]

    config = {'audio_path': '../../MAPS',
              'train_fold': './splits/{}/train'.format(split),
              'valid_fold': './splits/{}/valid'.format(split),
              'test_fold': './splits/{}/test'.format(split),
              'tfrecords_train_fold': './tfrecords-dataset/{}/train/'.format(split),
              'tfrecords_valid_fold': './tfrecords-dataset/{}/valid/'.format(split),
              'tfrecords_test_fold': './tfrecords-dataset/{}/test/'.format(split),
              'chord_folder': './tfrecords-dataset/chords/',
              'chord_fold': './splits/chord-splits/train',
              'chroma_folder': './chroma/',
              'context_frames': 2000,
              'is_chroma': False,
              'is_hpcp': False,
              'audio_config': {'num_channels': 1,
                               'sample_rate': 44100,
                               'filterbank': 'LogarithmicFilterbank',
                               'frame_size': 4096,
                               'fft_size': 4096,
                               'fps': 100,
                               'num_bands': 48,
                               'fmin': 10.0, # 30
                               'fmax': 5000.0, # 8000
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
              'fft_size': 4096*3,
              'fps': 100,
              'num_classes': 12,
              # dont use lowest piano octave, since the spectrogram does not provide enough information in the low freq
              'fmin': [27.5, 54.0, 107.0, 215.0, 426.0, 856.0, 1701.0, 3423.0],
              'fmax': [53.0, 106.0, 214.0, 425.0, 855.0, 1700.0, 3422.0, 6644.9],
              'fref': 440.0,
              'window': 1,
              'norm_filters': False,
              'circular_shift': False,
              'norm': True}
    return config
