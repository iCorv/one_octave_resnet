import numpy as np
from numpy import loadtxt
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

def midi_to_hz(midi_num, fref=440.0):
    return np.float_power(2, ((midi_num-69)/12)) * fref

sorted_ground_truth_list = glob.glob('/Users/Jaedicke/MAPS/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht.txt')
data = np.load("notes_2018-12-10.npz")
note_map = data["notes"]
print(np.shape(note_map))
num_est_notes = np.sum(np.sum(note_map), dtype=np.int64)
print(num_est_notes)

# load ground truth
ground_truth = loadtxt(sorted_ground_truth_list[0], skiprows=1, delimiter='\t')

ref_intervals = ground_truth[:, 0:2]
ref_pitches = midi_to_hz(ground_truth[:, 2])

est_intervals = np.zeros((num_est_notes, 2))
#est_pitches = np.zeros((num_est_notes, 1))


notes_index = np.where(note_map)

est_pitches = midi_to_hz(notes_index[0] + 21)
est_intervals[:, 0] = (notes_index[1] * 0.01) + (frame_size/sample_rate * 0) + frame_size/sample_rate/2
est_intervals[:, 1] = (notes_index[1] * 0.01) + (frame_size/sample_rate * 0) + frame_size/sample_rate

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