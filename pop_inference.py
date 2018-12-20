import tensorflow as tf
import pop_model
import configurations.pop_hyper_parameters as php
import numpy as np
import pop_preprocessing as prep
import configurations.pop_preprocessing_parameters as ppp
from scipy.io import savemat


def convert_fold_to_note_activation(fold, mode, net, model_dir, norm=False):
    """Preprocess an entire fold as defined in the preprocessing parameters and classify its note activations.
        fold - Fold.fold_1, Fold.fold_2, Fold.fold_3, Fold.fold_4, Fold.fold_benchmark
        mode - 'train', 'valid' or 'test' to address the correct config parameter
        norm - Flag if the spectrogram should be normed to 1
        net - The network used for classification, e.g. 'ConvNet, 'ResNet_v1'
        model_dir - e.g. "./model_ResNet_fold_4/model.ckpt-1477730" for a specific checkpoint
    """
    config = ppp.get_preprocessing_parameters(fold.value)
    audio_config = config['audio_config']

    # load fold
    filenames = open(config[mode+'_fold'], 'r').readlines()
    filenames = [f.strip() for f in filenames]

    predictor = build_predictor(net, model_dir)

    for file in filenames:
        # split file path string at "/" and take the last split, since it's the actual filename
        write_note_activation_to_mat(config['note_activation_folder'] + file.split('/')[-1], config['audio_path'], file,
                                     audio_config, norm, config['context_frames'], predictor, config['is_chroma'])


def write_note_activation_to_mat(write_file, base_dir, read_file, audio_config, norm, context_frames, predictor, is_chroma):
    """Transforms a wav and mid file to features and writes them to a tfrecords file."""
    spectrogram = prep.wav_to_spec(base_dir, read_file, audio_config)
    ground_truth = prep.midi_to_groundtruth(base_dir, read_file, 1. / audio_config['fps'], spectrogram.shape[0], is_chroma)
    # re-scale spectrogram to the range [0, 1]
    if norm:
        spectrogram = np.divide(spectrogram, np.max(spectrogram))
    note_activation = spectrogram_to_note_activation(spectrogram, context_frames, predictor)

    savemat(write_file, {"features": note_activation, "labels": ground_truth})


def convert_fold_to_chroma(fold, mode, norm=False):
    """Preprocess an entire fold as defined in the preprocessing parameters and classify its chroma
        fold - Fold.fold_1, Fold.fold_2, Fold.fold_3, Fold.fold_4, Fold.fold_benchmark
        mode - 'train', 'valid' or 'test' to address the correct config parameter
    """
    config = ppp.get_preprocessing_parameters(fold.value)
    audio_config = config['audio_config']

    # load fold
    filenames = open(config[mode+'_fold'], 'r').readlines()
    filenames = [f.strip() for f in filenames]

    predictor = build_predictor()

    for file in filenames:
        # split file path string at "/" and take the last split, since it's the actual filename
        write_chroma_to_npz(config['chroma_folder'] + file.split('/')[-1], config['audio_path'], file, audio_config,
                            norm, config['context_frames'], predictor)


def write_chroma_to_npz(write_file, base_dir, read_file, audio_config, norm, context_frames, predictor):
    """Transforms a wav and mid file to features and writes them to a tfrecords file."""
    spectrogram = prep.wav_to_spec(base_dir, read_file, audio_config)
    # re-scale spectrogram to the range [0, 1]
    if norm:
        spectrogram = np.divide(spectrogram, np.max(spectrogram))
    chroma = spectrogram_to_chroma(spectrogram, context_frames, predictor)

    np.savez(write_file, chroma=chroma)


def serving_input_fn():
    features = tf.placeholder(dtype=tf.float32, shape=[5, 229], name='features')
    return tf.estimator.export.TensorServingInputReceiver(features, features)


def build_predictor(net, model_dir):
    hparams = php.get_hyper_parameters(net)
    classifier = tf.estimator.Estimator(
        model_fn=pop_model.conv_net_model_fn,
        model_dir=model_dir,
        #warm_start_from=model_dir,
        params=hparams)

    estimator_predictor = tf.contrib.predictor.from_estimator(classifier, serving_input_fn, output_key='predictions', checkpoint_path="./model_ResNet_fold_4/model.ckpt-1477730")
    return estimator_predictor


def get_activation(features, estimator_predictor):
    p = estimator_predictor({'input': features})

    return p['probabilities']


def spectrogram_to_chroma(spec, context_frames, estimator_predictor):
    chroma = np.zeros([spec.shape[0], 12])
    for frame in range(context_frames, spec.shape[0] - context_frames):
        chroma[frame, :] = get_activation(spec[frame - context_frames:frame + context_frames + 1, :], estimator_predictor)
    log_chroma = np.log(1.0 + chroma)
    return log_chroma/np.max(log_chroma)


def spectrogram_to_note_activation(spec, context_frames, estimator_predictor):
    note_activation = np.zeros([spec.shape[0], 88])
    for frame in range(context_frames, spec.shape[0] - context_frames):
        note_activation[frame, :] = get_activation(spec[frame - context_frames:frame + context_frames + 1, :],
                                                   estimator_predictor)
    return note_activation
