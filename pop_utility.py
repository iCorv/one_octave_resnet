"""Utility methods used in the project

"""
import numpy as np


def find_onset_frame(onset_in_sec, hop_size, sample_rate):
    """Computes the frame were the onset is nearest to the start of the frame."""
    frame_onset_in_samples = onset_in_sec * sample_rate
    onset_in_frame = frame_onset_in_samples / hop_size

    return int(round(onset_in_frame))


def midi_to_hz(midi_num, fref=440.0):
    """Transform a midi note to Herz."""
    return np.float_power(2, ((midi_num-69)/12)) * fref


def eval_frame_wise(predictions, targets, thresh=0.5):
    """
    """
    if predictions.shape != targets.shape:
        raise ValueError('predictions.shape {} != targets.shape {} !'.format(predictions.shape, targets.shape))

    pred = predictions > thresh
    targ = targets > thresh

    tp = pred & targ
    fp = pred ^ tp
    fn = targ ^ tp

    # tp, fp, tn, fn
    return prf_framewise(tp.sum(), fp.sum(), 0, fn.sum())


def prf_framewise(tp, fp, tn, fn):
    tp, fp, tn, fn = float(tp), float(fp), float(tn), float(fn)

    if tp + fp == 0.:
        p = 0.
    else:
        p = tp / (tp + fp)

    if tp + fn == 0.:
        r = 0.
    else:
        r = tp / (tp + fn)

    if p + r == 0.:
        f = 0.
    else:
        f = 2 * ((p * r) / (p + r))

    if tp + fp + fn == 0.:
        a = 0.
    else:
        a = tp / (tp + fp + fn)

    return p, r, f, a


def mean_eval_frame_wise(frame_wise_metrics, num_pieces):
    mean_frame_wise = (sum([f[0] for f in frame_wise_metrics]) / num_pieces,
                       sum([f[1] for f in frame_wise_metrics]) / num_pieces,
                       sum([f[2] for f in frame_wise_metrics]) / num_pieces)
    return mean_frame_wise


def var_eval_frame_wise(frame_wise_metrics, mean_frame_wise, num_pieces):
    var_frame_wise = (sum([(f[0] - mean_frame_wise[0]) ** 2 for f in frame_wise_metrics]) / num_pieces,
                      sum([(f[1] - mean_frame_wise[1]) ** 2 for f in frame_wise_metrics]) / num_pieces,
                      sum([(f[2] - mean_frame_wise[2]) ** 2 for f in frame_wise_metrics]) / num_pieces)
    return var_frame_wise
