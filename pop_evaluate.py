import numpy as np
from numpy import loadtxt
from scipy import signal
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import librosa
import madmom
import glob
import mir_eval.transcription as tr

def reduce_consecutive_ones(arr, arr_length):
    reduced_arr = np.zeros(arr.shape)
    found_start = False
    start = 0
    for index in range(0, arr_length):
        if arr[index] == 1 and not found_start:
            start = index
            found_start = True
        if arr[index] == 0 and found_start:
            center = start + int(np.floor((index-1 - start)/2))
            reduced_arr[center] = 1
            found_start = False
    return reduced_arr

def reduce_consecutive_ones_mat(mat, mat_length):
    for index in range(0, 88):
        mat[index, :] = reduce_consecutive_ones(mat[index, :], mat_length)
    return mat


# this function finds the frame were the onset is nearest to the middle of the frame
def find_onset_frame(onset_in_sec, frame_length, hop_size, sample_rate):
    frame_onset_in_samples = onset_in_sec * sample_rate - (frame_length / 2)
    onset_in_frame = frame_onset_in_samples / hop_size

    return np.round(onset_in_frame, decimals=0).astype(int)


def piano_roll_rep(onset_frames, midi_pitches, piano_roll_shape):
    piano_roll = np.zeros(piano_roll_shape)
    piano_roll[midi_pitches, onset_frames] = 1
    return piano_roll


def eval_framewise(predictions, targets, thresh=0.5):
    """
    author: filip (+ data-format amendments by rainer)
    """
    if predictions.shape != targets.shape:
        raise ValueError('predictions.shape {} != targets.shape {} !'.format(predictions.shape, targets.shape))

    pred = predictions > thresh
    targ = targets > thresh

    tp = pred & targ
    fp = pred ^ tp
    fn = targ ^ tp

    # tp, fp, tn, fn
    return tp.sum(), fp.sum(), 0, fn.sum()


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


hop_size = 0.01
sample_rate = 22050
frame_size = 512

midi_range_low = 21
midi_range_high = 108
midi_range = np.arange(midi_range_low, midi_range_high+1)

hamming = np.hamming(5)

def midi_to_hz(midi_num, fref=440.0):
    return np.float_power(2, ((midi_num-69)/12)) * fref

sorted_ground_truth_list = glob.glob('/Users/Jaedicke/MAPS_real_piano/ENSTDkAm/MUS/MAPS_MUS-chpn_op7_1_ENSTDkAm.txt')
#data = np.load("props_MAPS_MUS-chpn_op7_1_ENSTDkAm_2018-23-10.npz")
data = np.load("props_MAPS_MUS-chpn_op7_1_ENSTDkAm_2018-05-11.npz")

#act = madmom.features.notes.RNNPianoNoteProcessor()('/Users/Jaedicke/MAPS_real_piano/ENSTDkAm/MUS/MAPS_MUS-chpn_op7_1_ENSTDkAm.wav')

#props = signal.convolve2d(data["props"], hamming, mode='same')
props = data["props"]
prefix = np.zeros((88, 4))
props = np.append(prefix, props, axis=1)


fps = 1/hop_size
proc = madmom.features.notes.NotePeakPickingProcessor(threshold=0.5, pre_max=1.0/fps, post_max=1.0/fps, delay=-0.13, combine=0.03, smooth=0.3, fps=fps)

est_intervals_notes = proc(props.T)
#est_intervals_notes = proc(act)



#for bins in range(0, 87):
#    props[bins, :] = signal.convolve(props[bins, :], hamming, mode='same')
#note_map = np.where(props > 0.7, 1, 0)
#note_map_shape = note_map.shape
#note_map = reduce_consecutive_ones_mat(note_map, note_map_shape[1])
#note_map = data["notes"]



num_est_notes = np.shape(est_intervals_notes)[0]
print("number of estimated notes: " + str(num_est_notes))

# load ground truth
ground_truth = loadtxt(sorted_ground_truth_list[0], skiprows=1, delimiter='\t')

# calculate targets
onset_frames = find_onset_frame(ground_truth[:, 0], frame_length=frame_size, hop_size=librosa.time_to_samples(0.01, sr=sample_rate), sample_rate=sample_rate)
pitch_per_frame = ground_truth[:, 2] - midi_range_low

targets = piano_roll_rep(onset_frames=onset_frames, midi_pitches=pitch_per_frame.astype(int), piano_roll_shape=np.shape(props))

tp, fp, tn, fn = eval_framewise(props, targets)

p, r, f, a = prf_framewise(tp, fp, tn, fn)

print("TP: " + str(tp))
print("FP: " + str(fp))
print("TN: " + str(tn))
print("FN: " + str(fn))
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1-Score: " + str(f))
print("Accuracy: " + str(a))


# find values within range
midi_range_bool = np.isin(ground_truth[:, 2], midi_range)

midi_range_indices = np.where(midi_range_bool)
print("number of notes in piece: " + str(np.size(midi_range_indices)))

ref_intervals = np.squeeze(ground_truth[midi_range_indices, 0:2])
ref_pitches = midi_to_hz(np.squeeze(ground_truth[midi_range_indices, 2]))

est_intervals = np.zeros((num_est_notes, 2))



#notes_index = np.where(note_map)


#est_pitches = midi_to_hz(notes_index[0] + midi_range_low)
#est_intervals[:, 0] = (notes_index[1] * 0.01) + frame_size/sample_rate/2
#est_intervals[:, 1] = (notes_index[1] * 0.01) + frame_size/sample_rate

est_pitches = midi_to_hz(est_intervals_notes[:, 1])
print(est_intervals_notes[:, 1])
est_intervals[:, 0] = est_intervals_notes[:, 0]
est_intervals[:, 1] = est_intervals_notes[:, 0] + frame_size/sample_rate

metrics_with_pitch = tr.precision_recall_f1_overlap(ref_intervals, ref_pitches, est_intervals,
                                                    est_pitches, onset_tolerance=0.05,
                                                    pitch_tolerance=50.0, offset_ratio=None,
                                                    strict=False,
                                                    beta=1.0)

metrics_onset = tr.onset_precision_recall_f1(ref_intervals, est_intervals,
                                             onset_tolerance=0.05, strict=False, beta=1.0)

# matched_notes = tr.match_notes(ref_intervals, ref_pitches, est_intervals, est_pitches,
#                                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
#                                offset_min_tolerance=0.05, strict=False)

print(metrics_onset)
print(metrics_with_pitch)
#print(matched_notes)