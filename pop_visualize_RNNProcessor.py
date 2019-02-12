import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import madmom
from numpy import loadtxt
from scipy import signal
import pypianoroll as ppr
from pypianoroll import Multitrack, Track
import glob
import librosa

sr = 22050
hop_size = librosa.time_to_samples(0.01, sr=sr)
hop_size_sec = 0.01
frame_length = 512

plot_frames = 500

midi_range_low = 21
midi_range_high = 108
midi_range = np.arange(midi_range_low, midi_range_high+1)

# this function finds the frame were the onset is nearest to the middle of the frame
def find_onset_frame(onset_in_sec, frame_length, hop_size, sample_rate):
    frame_onset_in_samples = onset_in_sec * sample_rate - (frame_length / 2)
    onset_in_frame = frame_onset_in_samples / hop_size

    return np.round(onset_in_frame, decimals=0).astype(int)


def piano_roll_rep(onset_frames, midi_pitches, piano_roll_shape):
    piano_roll = np.zeros(piano_roll_shape)
    piano_roll[midi_pitches, onset_frames] = 1
    return piano_roll


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


props = madmom.features.notes.RNNPianoNoteProcessor()('/Users/Jaedicke/MAPS_real_piano/ENSTDkAm/MUS/MAPS_MUS-chpn_op7_1_ENSTDkAm.wav')
props = props.T

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)


ax1.pcolormesh((props[:, 0:plot_frames]))
ax1.set_title("Activation Function")
locs, labels = plt.yticks()
#plt.yticks(locs, np.arange(21, 108, 1))
plt.grid(True)

#props = signal.convolve2d(data["props"], hamming, mode='same')



fps = 1/hop_size_sec
proc = madmom.features.notes.NotePeakPickingProcessor(threshold=0.5, pre_max=1.0/fps, post_max=1.0/fps, delay=-0.0, combine=0.03, smooth=0.0, fps=fps)

est_intervals_notes = proc(props.T)

est_onset_frames = find_onset_frame(est_intervals_notes[:, 0], frame_length=frame_length, hop_size=hop_size, sample_rate=sr)

est_piano_roll = piano_roll_rep(onset_frames=est_onset_frames, midi_pitches=est_intervals_notes[:, 1].astype(int)-21, piano_roll_shape=np.shape(props))

ax2.pcolormesh((est_piano_roll[:, 0:plot_frames]))
ax2.set_title("Peak Picked Notes")

#locs, labels = plt.yticks()
# set last y-tick to 88
#locs[-1] = 88
# find labels in frequency bins
#plt.yticks(locs, np.append(freq_bins.round(decimals=1)[::10], freq_bins.round(decimals=1)[-1]))
#plt.title(title)
#ax1.colorbar(format='%+2.0f dB')
#plt.tight_layout()
#plt.show()


sorted_ground_truth_list = glob.glob('/Users/Jaedicke/MAPS_real_piano/ENSTDkAm/MUS/MAPS_MUS-chpn_op7_1_ENSTDkAm.txt')
# load ground truth
ground_truth = loadtxt(sorted_ground_truth_list[0], skiprows=1, delimiter='\t')
# find values within range
midi_range_bool = np.isin(ground_truth[:, 2], midi_range)
midi_range_indices = np.where(midi_range_bool)


onset_frames = find_onset_frame(ground_truth[midi_range_indices, 0], frame_length=frame_length, hop_size=hop_size, sample_rate=sr)
pitch_per_frame = ground_truth[midi_range_indices, 2] - midi_range_low


piano_roll = piano_roll_rep(onset_frames=onset_frames, midi_pitches=pitch_per_frame.astype(int), piano_roll_shape=np.shape(props))



ax3.pcolormesh((piano_roll[:, 0:plot_frames]))
ax3.set_title("Ground Truth")


#pianoroll = ppr.parse("/Users/Jaedicke/Desktop/MAPS_MUS-alb_se3_AkPnBcht.mid", beat_resolution=24,
#          name='MAPS_MUS-alb_se3_AkPnBcht')
#pianoroll = Multitrack('/Users/Jaedicke/Desktop/MAPS_MUS-alb_se3_AkPnBcht.mid')
#ppr.plot(pianoroll, filename=None, mode='separate', track_label='name', preset='frame', cmaps=None, xtick='auto', ytick='octave', xticklabel=True, yticklabel='auto', tick_loc=None, tick_direction='in', label='both', grid='off', grid_linestyle=':', grid_linewidth=0.5)


plt.show()
