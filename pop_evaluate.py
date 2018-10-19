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
data = np.load("props_MAPS_MUS-chpn_op7_1_ENSTDkAm_2018-18-10.npz")

#props = signal.convolve2d(data["props"], hamming, mode='same')
props = data["props"]
prefix = np.zeros((88, 7))
props = np.append(prefix,props,axis=1)
#for bins in range(0, 87):
#    props[bins, :] = signal.convolve(props[bins, :], hamming, mode='same')
note_map = np.where(props > 0.5, 1, 0)
note_map_shape = note_map.shape
note_map = reduce_consecutive_ones_mat(note_map, note_map_shape[1])
#note_map = data["notes"]
print(np.shape(note_map))
num_est_notes = np.sum(np.sum(note_map), dtype=np.int64)
print(num_est_notes)

# load ground truth
ground_truth = loadtxt(sorted_ground_truth_list[0], skiprows=1, delimiter='\t')


# find values within range
midi_range_bool = np.isin(ground_truth[:, 2], midi_range)

midi_range_indices = np.where(midi_range_bool)
print(np.size(midi_range_indices))

ref_intervals = np.squeeze(ground_truth[midi_range_indices, 0:2])
ref_pitches = midi_to_hz(np.squeeze(ground_truth[midi_range_indices, 2]))

est_intervals = np.zeros((num_est_notes, 2))



notes_index = np.where(note_map)


est_pitches = midi_to_hz(notes_index[0] + midi_range_low)
est_intervals[:, 0] = (notes_index[1] * 0.01) + frame_size/sample_rate/2
est_intervals[:, 1] = (notes_index[1] * 0.01) + frame_size/sample_rate

metrics_with_pitch = tr.precision_recall_f1_overlap(ref_intervals, ref_pitches, est_intervals,
                                                    est_pitches, onset_tolerance=0.3,
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