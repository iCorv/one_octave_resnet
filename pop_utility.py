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


def piano_roll_rep(onset_frames, midi_pitches, piano_roll_shape, onset_duration=None):
    piano_roll = np.zeros(piano_roll_shape)
    if onset_duration is not None:
        for onset_frame, midi_pitch in list(zip(onset_frames, midi_pitches)):
            piano_roll[onset_frame:onset_frame+onset_duration, midi_pitch] = 1
    else:
        piano_roll[onset_frames, midi_pitches] = 1
    return piano_roll


def pianoroll_to_interval_sequence(frames,
                                   frames_per_second,
                                   min_midi_pitch=21,
                                   onset_predictions=None,
                                   offset_predictions=None):
    """Convert frames to an interval sequence."""
    frame_length_seconds = 1 / frames_per_second

    est_pitches = np.ndarray([1, ])
    est_intervals = np.ndarray([1, 2])

    pitch_start_step = {}

    # Add silent frame at the end so we can do a final loop and terminate any
    # notes that are still active.
    frames = np.append(frames, [np.zeros(frames[0].shape)], 0)

    if onset_predictions is not None:
        onset_predictions = np.append(onset_predictions, [np.zeros(onset_predictions[0].shape)], 0)
        # Ensure that any frame with an onset prediction is considered active.
        frames = np.logical_or(frames, onset_predictions)

    elif offset_predictions is not None:
        offset_predictions = np.append(offset_predictions,
                                       [np.zeros(offset_predictions[0].shape)], 0)
        # If the frame and offset are both on, then turn it off
        frames[np.where(np.logical_and(frames > 0, offset_predictions > 0))] = 0

    def end_pitch(pitch, end_frame):
        """End an active pitch."""
        start_time = pitch_start_step[pitch] * frame_length_seconds
        end_time = end_frame * frame_length_seconds

        if (end_time - start_time) * 1000 >= frame_length_seconds:
            e_intervals = [[start_time, end_time]]
            e_pitches = [pitch + min_midi_pitch]
        else:
            e_intervals = None
            e_pitches = None

        del pitch_start_step[pitch]

        return e_intervals, e_pitches

    # def process_active_pitch(pitch, i):
    #     """Process a pitch being active in a given frame."""
    #     if pitch not in pitch_start_step:
    #         pitch_start_step[pitch] = i

    def process_active_pitch(pitch, i):
        """Process a pitch being active in a given frame."""
        if pitch not in pitch_start_step:
            if onset_predictions is not None:
                # If onset predictions were supplied, only allow a new note to start
                # if we've predicted an onset.
                if onset_predictions[i, pitch]:
                    pitch_start_step[pitch] = i
                else:
                    # Even though the frame is active, the onset predictor doesn't
                    # say there should be an onset, so ignore it.
                    pass
            else:
                pitch_start_step[pitch] = i
        else:
            if onset_predictions is not None:
                # pitch is already active, but if this is a new onset, we should end
                # the note and start a new one.
                if (onset_predictions[i, pitch] and
                        not onset_predictions[i - 1, pitch]):
                    end_pitch(pitch, i)
                    pitch_start_step[pitch] = i

    for i, frame in enumerate(frames):
        for pitch, active in enumerate(frame):
            if active:
                process_active_pitch(pitch, i)
            elif pitch in pitch_start_step:
                est_i, est_p = end_pitch(pitch, i)
                if est_i is not None:
                    est_pitches = np.append(est_pitches, est_p, axis=0)
                    est_intervals = np.append(est_intervals, est_i, axis=0)

    # remove first default entry
    est_pitches = est_pitches[1:]
    est_intervals = est_intervals[1:]

    total_time = len(frames) * frame_length_seconds
    # make sure last note ends before end of piece
    if est_pitches.size is not 0:
        assert total_time >= est_intervals[-1, -1]

    # return intervals, pitches
    return est_intervals, est_pitches

