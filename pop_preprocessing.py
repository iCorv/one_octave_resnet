"""Shared methods to provide data in the tfrecords format to the transcription model.

    frame: An individual row of a spectrogram computed from some
           number of audio samples.
    example: An individual training example. The number of frames in an example
             is determined by the context length, before and after the center frame.

    Some methods are adapted from:
    @inproceedings{kelz_icassp18, author = {Rainer Kelz and Gerhard Widmer},
    title = {Investigating Label Noise Sensitivity Of Convolutional Neural Networks For Fine Grained Audio Signal Labelling}
    booktitle = {2018 {IEEE} International Conference on Acoustics, Speech and Signal Processing, {ICASSP} 2018, Calgary,
    Alberta, Canada, April 15-20, 2018}, year = {2018} }
    https://github.com/rainerkelz/ICASSP18
"""

import numpy as np
import madmom
import tensorflow as tf
import os
import configurations.pop_preprocessing_parameters as ppp
import warnings
from joblib import Parallel, delayed
import multiprocessing
from madmom.utils import midi
from enum import Enum
warnings.filterwarnings("ignore")


class Fold(Enum):
    """Distinguish the different folds the model is trained on."""
    fold_1 = 0
    fold_2 = 1
    fold_3 = 2
    fold_4 = 3
    fold_benchmark = 4
    fold_single_note = 5


def wav_to_spec(base_dir, filename, _audio_options):
    """Transforms the contents of a wav file into a series of spec frames."""
    audio_filename = os.path.join(base_dir, filename + '.wav')

    spec_type, audio_options = get_spec_processor(_audio_options, madmom.audio.spectrogram)

    # it's necessary to cast this to np.array, b/c the madmom-class holds references to way too much memory
    spectrogram = np.array(spec_type(audio_filename, **audio_options))
    return spectrogram


def get_spec_processor(_audio_options, madmom_spec):
    """Returns the madmom spectrogram processor as defined in audio options."""
    audio_options = dict(_audio_options)

    if 'spectrogram_type' in audio_options:
        spectype = getattr(madmom_spec, audio_options['spectrogram_type'])
        del audio_options['spectrogram_type']
    else:
        spectype = getattr(madmom_spec, 'LogarithmicFilteredSpectrogram')

    if 'filterbank' in audio_options:
        audio_options['filterbank'] = getattr(madmom_spec, audio_options['filterbank'])
    else:
        audio_options['filterbank'] = getattr(madmom_spec, 'LogarithmicFilterbank')

    return spectype, audio_options


def midi_to_groundtruth(base_dir, filename, dt, n_frames):
    """Computes the frame-wise ground truth from a midi file."""
    midi_filename = os.path.join(base_dir, filename + '.mid')
    pattern = midi.MIDIFile.from_file(midi_filename)
    ground_truth = np.zeros((n_frames, 88)).astype(np.int64)
    for onset, _pitch, duration, velocity, _channel in pattern.notes():
        pitch = int(_pitch)
        frame_start = int(np.round(onset / dt))
        frame_end = int(np.round((onset + duration) / dt))
        label = pitch - 21
        ground_truth[frame_start:frame_end, label] = 1
    return ground_truth


def _float_feature(value):
    """Converts a value to a tensorflow feature for float data types."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Converts a value to a tensorflow feature for int data types."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Converts a value to a tensorflow feature for byte data types."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def preprocess_fold(fold, mode, norm=False):
    """Preprocess an entire fold as defined in the preprocessing parameters.
        fold - Fold.fold_1, Fold.fold_2, Fold.fold_3, Fold.fold_4, Fold.fold_benchmark
        mode - 'train', 'valid' or 'test' to address the correct config parameter
    """
    config = ppp.get_preprocessing_parameters(fold.value)
    audio_config = config['audio_config']

    # load fold
    filenames = open(config[mode+'_fold'], 'r').readlines()
    filenames = [f.strip() for f in filenames]

    total_examples_processed = 0

    for file in filenames:
        # split file path string at "/" and take the last split, since it's the actual filename
        num_ex_processed = write_file_to_tfrecords(config['tfrecords_'+mode+'_fold'] + file.split('/')[-1] +
                                                   ".tfrecords", config['audio_path'], file, audio_config, norm,
                                                   config['context_frames'])
        total_examples_processed = total_examples_processed + num_ex_processed

    print("Examples processed: " + str(total_examples_processed))
    np.savez(config['tfrecords_' + mode + '_fold'] + "total_examples_processed",
             total_examples_processed=total_examples_processed)


def preprocess_fold_parallel(fold, mode, norm=False):
    """Parallel preprocess an entire fold as defined in the preprocessing parameters.
        This seems only to work on Win with Anaconda!
        fold - Fold.fold_1, Fold.fold_2, Fold.fold_3, Fold.fold_4, Fold.fold_benchmark
        mode - 'train', 'valid' or 'test' to address the correct config parameter
    """
    config = ppp.get_preprocessing_parameters(fold.value)
    audio_config = config['audio_config']

    # load fold
    filenames = open(config[mode+'_fold'], 'r').readlines()
    filenames = [f.strip() for f in filenames]

    def parallel_loop(file):
        # split file path string at "/" and take the last split, since it's the actual filename
        num_ex_processed = write_file_to_tfrecords(config['tfrecords_'+mode+'_fold'] + file.split('/')[-1] +
                                                   ".tfrecords", config['audio_path'], file, audio_config, norm,
                                                   config['context_frames'])
        return num_ex_processed

    num_cores = multiprocessing.cpu_count()

    total_examples_processed = Parallel(n_jobs=num_cores)(delayed(parallel_loop)(file) for file in filenames)
    print("Examples processed: " + str(np.sum(total_examples_processed)))
    np.savez(config['tfrecords_' + mode + '_fold'] + "total_examples_processed",
             total_examples_processed=np.sum(total_examples_processed))


def write_file_to_tfrecords(write_file, base_dir, read_file, audio_config, norm, context_frames):
    """Transforms a wav and mid file to features and writes them to a tfrecords file."""
    writer = tf.python_io.TFRecordWriter(write_file)
    spectrogram = wav_to_spec(base_dir, read_file, audio_config)
    ground_truth = midi_to_groundtruth(base_dir, read_file, 1. / audio_config['fps'], spectrogram.shape[0])
    total_examples_processed = 0
    # re-scale spectrogram to the range [0, 1]
    if norm:
        spectrogram = np.divide(spectrogram, np.max(spectrogram))

    for frame in range(context_frames, spectrogram.shape[0] - context_frames):
        example = features_to_example(spectrogram[frame - context_frames:frame + context_frames + 1, :],
                                      ground_truth[frame, :])

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        total_examples_processed = total_examples_processed + 1

    writer.close()
    return total_examples_processed


def features_to_example(spectrogram, ground_truth):
    """Build an example from spectrogram and ground truth data."""
    # Create a feature
    feature = {"label": _int64_feature(ground_truth),
               "spec": _float_feature(spectrogram.ravel())}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example
