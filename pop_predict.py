import tensorflow as tf
import pop_conv_net_kelz
import configurations.pop_hyper_parameters as php
import numpy as np

hparams = php.get_hyper_parameters('ConvNet')


def serving_input_fn():
    features = tf.placeholder(dtype=tf.float32, shape=[5, 229], name='features')
    return tf.estimator.export.TensorServingInputReceiver(features, features)


def build_predictor():
    classifier = tf.estimator.Estimator(
        model_fn=pop_conv_net_kelz.conv_net_model_fn,
        model_dir="./model_kelz_chroma",
        #warm_start_from="./model/model.ckpt-1323466",
        params=hparams)

    estimator_predictor = tf.contrib.predictor.from_estimator(classifier, serving_input_fn, output_key='predictions')
    return estimator_predictor


def get_activation(features, estimator_predictor):

    p = estimator_predictor({'input': features})
    #print(p['probabilities'].shape)
    return p['probabilities']


def spectrogram_to_chroma(spec, context_frames, estimator_predictor):
    chroma = np.zeros([spec.shape[0], 12])
    for frame in range(context_frames, spec.shape[0] - context_frames):
        chroma[frame, :] = get_activation(spec[frame - context_frames:frame + context_frames + 1, :], estimator_predictor)
    log_chroma = np.log(1.0 + chroma)
    return log_chroma/np.max(log_chroma)
    #return chroma


