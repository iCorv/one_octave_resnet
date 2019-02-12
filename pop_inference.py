import tensorflow as tf
import pop_model
import configurations.pop_hyper_parameters as php
import numpy as np
import pop_preprocessing as prep
import configurations.pop_preprocessing_parameters as ppp
import pop_utility as util
import madmom
from scipy.io import savemat


def convert_fold_to_note_activation(fold, mode, net, model_dir, save_dir, norm=False):
    """Preprocess an entire fold as defined in the preprocessing parameters and classify its note activations.
        fold - Fold.fold_1, Fold.fold_2, Fold.fold_3, Fold.fold_4, Fold.fold_benchmark
        mode - 'train', 'valid' or 'test' to address the correct config parameter
        norm - Flag if the spectrogram should be normed to 1
        net - The network used for classification, e.g. 'ConvNet, 'ResNet_v1'
        model_dir - e.g. "./model_ResNet_fold_4/". For a specific checkpoint, change the checkpoint number in the
        chekpoint file from the model folder.
        save_dir - folder were to save note activations, e.g. "./note_activations/"
    """
    config = ppp.get_preprocessing_parameters(fold.value)
    audio_config = config['audio_config']

    # load fold
    filenames = open(config[mode+'_fold'], 'r').readlines()
    filenames = [f.strip() for f in filenames]

    predictor = build_predictor(net, model_dir)

    for file in filenames:
        # split file path string at "/" and take the last split, since it's the actual filename
        write_note_activation_to_mat(save_dir + file.split('/')[-1], config['audio_path'], file,
                                     audio_config, norm, config['context_frames'], predictor)


def write_note_activation_to_mat(write_file, base_dir, read_file, audio_config, norm, context_frames, predictor):
    """Transforms a wav and mid file to features and writes them to a tfrecords file."""
    spectrogram = prep.wav_to_spec(base_dir, read_file, audio_config)
    ground_truth = prep.midi_to_groundtruth(base_dir, read_file, 1. / audio_config['fps'], spectrogram.shape[0])
    # re-scale spectrogram to the range [0, 1]
    if norm:
        spectrogram = np.divide(spectrogram, np.max(spectrogram))
    note_activation = spectrogram_to_note_activation(spectrogram, context_frames, predictor)

    savemat(write_file, {"features": note_activation, "labels": ground_truth})

def get_note_activation(base_dir, read_file, audio_config, norm, context_frames, predictor):
    """Transforms a wav and mid file to features and writes them to a tfrecords file."""
    spectrogram = prep.wav_to_spec(base_dir, read_file, audio_config)
    gt_frame, gt_onset, gt_offset = prep.midi_to_triple_groundtruth(base_dir, read_file, 1. / audio_config['fps'], spectrogram.shape[0])
    # re-scale spectrogram to the range [0, 1]
    if norm:
        spectrogram = np.divide(spectrogram, np.max(spectrogram))
    note_activation = spectrogram_to_note_activation(spectrogram, context_frames, predictor)
    return note_activation, gt_frame, gt_onset, gt_offset


def compute_all_error_metrics(fold, mode, net, model_dir, save_dir, norm=False):
    """Error metrics for an entire fold as defined in the preprocessing parameters.
        fold - Fold.fold_1, Fold.fold_2, Fold.fold_3, Fold.fold_4, Fold.fold_benchmark
        mode - 'train', 'valid' or 'test' to address the correct config parameter
        norm - Flag if the spectrogram should be normed to 1
        net - The network used for classification, e.g. 'ConvNet, 'ResNet_v1'
        model_dir - e.g. "./model_ResNet_fold_4/". For a specific checkpoint, change the checkpoint number in the
        chekpoint file from the model folder.
        save_dir - folder were to save note activations, e.g. "./note_activations/"
    """
    config = ppp.get_preprocessing_parameters(fold.value)
    audio_config = config['audio_config']

    # load fold
    filenames = open(config[mode + '_fold'], 'r').readlines()
    filenames = [f.strip() for f in filenames]

    predictor = build_predictor(net, model_dir)
    frame_wise_metrics = []
    frame_wise_onset_metrics = []
    frame_wise_offset_metrics = []

    filenames = filenames[1:4]

    for file in filenames:
        # split file path string at "/" and take the last split, since it's the actual filename
        note_activation, gt_frame, gt_onset, gt_offset = get_note_activation(config['audio_path'], file, audio_config, norm, config['context_frames'], predictor)
        #p_frame, r_frame, f_frame, a_frame = util.eval_framewise(note_activation, gt_frame)
        frame_wise_metrics.append(util.eval_framewise(note_activation, gt_frame))
        # multiply note activation with ground truth in order to blend out the rest of the activation fn
        #p_onset, r_onset, f_onset, a_onset = util.eval_framewise(np.multiply(note_activation, gt_onset), gt_onset)
        frame_wise_onset_metrics.append(util.eval_framewise(np.multiply(note_activation, gt_onset), gt_onset))
        #p_offset, r_offset, f_offset, a_offset = util.eval_framewise(np.multiply(note_activation, gt_offset), gt_offset)
        frame_wise_offset_metrics.append(util.eval_framewise(np.multiply(note_activation, gt_offset), gt_offset))



        #print("frame F1: " + str(f_frame))
        #print("onset F1: " + str(f_onset))
        #print("offset F1: " + str(f_offset))
    mean_frame_wise_f = sum([f[0] for f in frame_wise_metrics])/len(frame_wise_metrics)
    var_frame_wise_f = sum([(f[0] - mean_frame_wise_f) ** 2 for f in frame_wise_metrics])/len(frame_wise_metrics)
    print(mean_frame_wise_f)
    print(var_frame_wise_f)



def get_serving_input_fn(frames, bins):
    def serving_input_fn():
        x = tf.placeholder(dtype=tf.float32, shape=[frames, bins], name='features')
        return tf.estimator.export.TensorServingInputReceiver(x, x)

    return serving_input_fn


def build_predictor(net, model_dir):
    hparams = php.get_hyper_parameters(net)
    classifier = tf.estimator.Estimator(
        model_fn=pop_model.conv_net_model_fn,
        model_dir=model_dir,
        # warm_start_from=model_dir,
        params=hparams)

    estimator_predictor = tf.contrib.predictor.from_estimator(classifier,
                                                              get_serving_input_fn(hparams['frames'],
                                                                                   hparams['freq_bins']),
                                                              output_key='predictions')
    return estimator_predictor


def get_activation(features, estimator_predictor):
    p = estimator_predictor({'input': features})

    return p['probabilities']


def spectrogram_to_note_activation(spec, context_frames, estimator_predictor):
    note_activation = np.zeros([spec.shape[0], 88])
    for frame in range(context_frames, spec.shape[0] - context_frames):
        note_activation[frame, :] = get_activation(spec[frame - context_frames:frame + context_frames + 1, :],
                                                   estimator_predictor)
    return note_activation


