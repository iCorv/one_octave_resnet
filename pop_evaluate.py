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

hop_size = 0.01
sample_rate = 22050
frame_size = 512

midi_range_low = 48
midi_range_high = 59
midi_range = np.arange(midi_range_low, midi_range_high+1)

hamming = np.matlib.repmat(np.hamming(5), 1, 88)

def midi_to_hz(midi_num, fref=440.0):
    return np.float_power(2, ((midi_num-69)/12)) * fref

sorted_ground_truth_list = glob.glob('/Users/Jaedicke/MAPS/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht.txt')
data = np.load("props_2018-15-10.npz")

props = signal.convolve2d(data["props"], hamming, mode='same')
#props = data["props"]
note_map = np.where(props > 50, 1, 0)
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
est_intervals[:, 0] = (notes_index[1] * 0.01) + (frame_size/sample_rate * 7) + frame_size/sample_rate/2
est_intervals[:, 1] = (notes_index[1] * 0.01) + (frame_size/sample_rate * 7) + frame_size/sample_rate

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