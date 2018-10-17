import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
from scipy import signal
import pypianoroll as ppr
from pypianoroll import Multitrack, Track
import glob
import librosa

sr = 22050
hop_size = hop_size = librosa.time_to_samples(0.01, sr=sr)
frame_length = 512

midi_range_low = 48
midi_range_high = 59
midi_range = np.arange(midi_range_low, midi_range_high+1)

# this function finds the frame were the onset is nearest to the middle of the frame
def find_onset_frame(onset_in_sec, frame_length, hop_size, sample_rate):
    frame_onset_in_samples = onset_in_sec * sample_rate - (frame_length / 2)
    onset_in_frame = frame_onset_in_samples / hop_size

    return np.round(onset_in_frame, decimals=0).astype(int)


hamming = np.matlib.repmat(np.hamming(5), 1, 88)

data = np.load("props_2018-15-10.npz")
props = signal.convolve2d(data["props"], hamming, mode='same')
#props = data["props"]
props = np.where(props > 40, 1, 0)
print(props.shape)
#props = np.round(props)
print(np.max(props))
print(np.min(props))
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
ax1.pcolormesh((props))
#locs, labels = plt.yticks()
# set last y-tick to 88
#locs[-1] = 88
# find labels in frequency bins
#plt.yticks(locs, np.append(freq_bins.round(decimals=1)[::10], freq_bins.round(decimals=1)[-1]))
#plt.title(title)
#ax1.colorbar(format='%+2.0f dB')
#plt.tight_layout()
#plt.show()


sorted_ground_truth_list = glob.glob('/Users/Jaedicke/MAPS/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht.txt')
# load ground truth
ground_truth = loadtxt(sorted_ground_truth_list[0], skiprows=1, delimiter='\t')
# find values within range
midi_range_bool = np.isin(ground_truth[:, 2], midi_range)
midi_range_indices = np.where(midi_range_bool)


onset_frames = find_onset_frame(ground_truth[midi_range_indices, 0], frame_length=frame_length, hop_size=hop_size, sample_rate=sr)
pitch_per_frame = ground_truth[midi_range_indices, 2] - midi_range_low



piano_roll = np.zeros(np.shape(props))
piano_roll[pitch_per_frame.astype(int), onset_frames] = 1


ax2.pcolormesh((piano_roll))


#pianoroll = ppr.parse("/Users/Jaedicke/Desktop/MAPS_MUS-alb_se3_AkPnBcht.mid", beat_resolution=24,
#          name='MAPS_MUS-alb_se3_AkPnBcht')
#pianoroll = Multitrack('/Users/Jaedicke/Desktop/MAPS_MUS-alb_se3_AkPnBcht.mid')
#ppr.plot(pianoroll, filename=None, mode='separate', track_label='name', preset='frame', cmaps=None, xtick='auto', ytick='octave', xticklabel=True, yticklabel='auto', tick_loc=None, tick_direction='in', label='both', grid='off', grid_linestyle=':', grid_linewidth=0.5)


plt.show()
