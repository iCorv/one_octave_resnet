import numpy as np
from numpy import loadtxt
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import librosa
import madmom
import glob
import tensorflow as tf


# this function finds the frame were the onset is nearest to the middle of the frame
def find_onset_frame(onset_in_sec, frame_length, hop_size, sample_rate):
    frame_onset_in_samples = onset_in_sec * sample_rate - (frame_length / 2)
    onset_in_frame = frame_onset_in_samples / hop_size

    return int(round(onset_in_frame))

def specshow(spec, title, center_frame, freq_bins):
    plt.figure()
    plt.pcolormesh(spec[(center_frame-7):(center_frame+8), :].T)
    locs, labels = plt.yticks()
    # set last y-tick to 88
    locs[-1] = 88
    # find labels in frequency bins
    plt.yticks(locs, np.append(freq_bins.round(decimals=1)[::10], freq_bins.round(decimals=1)[-1]))
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()


def midi_to_hz(midi_num, fref=440.0):
    return np.float_power(2, ((midi_num-69)/12)) * fref


def create_semitone_filterbank(frequencies):
    semitone_freq_bins = madmom.audio.filters.semitone_frequencies(fmin=27.5, fmax=4186.0090448096, fref=440.0)
    bins = madmom.audio.filters.frequencies2bins(frequencies, semitone_freq_bins, unique_bins=False)
    semitone_filterbank = np.zeros(shape=[88, len(bins)])
    current_bin = bins[0]
    start = 0
    for i in range (0, len(bins)):
        if bins[i] > current_bin:
            center = start + int(round((i-1 - start)/2))
            semitone_filterbank[current_bin][start:i] = madmom.audio.filters.TriangularFilter(start, center, i, norm=True)
            current_bin = bins[i]
            start = i
    # last semitone triangle filter
    center = start + int(round((len(bins) - 1 - start) / 2))
    semitone_filterbank[current_bin][start:len(bins)-1] = madmom.audio.filters.TriangularFilter(start, center, len(bins) - 1, norm=True)

    return semitone_filterbank.T, semitone_freq_bins


def create_mel_filterbank(frequencies):
    mel_freq_bins = madmom.audio.filters.mel_frequencies(num_bands=88, fmin=27.5, fmax=4186.0090448096)

    bins = madmom.audio.filters.frequencies2bins(frequencies, mel_freq_bins, unique_bins=False)

    mel_filterbank = np.zeros(shape=[88, len(bins)])
    current_bin = bins[0]
    start = 0
    for i in range (0, len(bins)):
        if bins[i] > current_bin:
            center = start + int(round((i-1 - start)/2))
            mel_filterbank[current_bin][start:i] = madmom.audio.filters.TriangularFilter(start, center, i, norm=True)
            current_bin = bins[i]
            start = i
    # last semitone triangle filter
    center = start + int(round((len(bins) - 1 - start) / 2))
    mel_filterbank[current_bin][start:len(bins)-1] = madmom.audio.filters.TriangularFilter(start, center, len(bins) - 1, norm=True)

    return mel_filterbank.T, mel_freq_bins


def count_examples(example_list):
    count = 0
    for example in range(0, len(example_list)):
        list_per_example = loadtxt(example_list[example], skiprows=1, delimiter='\t')
        if list_per_example.ndim == 1:
            num_per_example = 1
        else:
            num_per_example = list_per_example.shape[0]
        count = count + num_per_example
    return count


sr = 22050
frame_length_samples = [512, 1024, 2048]
frame_length_seconds = np.divide(frame_length_samples, sr)
num_frames = 15
hop_size = librosa.time_to_samples(0.01, sr=sr)
example_num = 1
frame_length_num = 0

sorted_audio_list = glob.glob('/Users/Jaedicke/MAPS/**/ISOL/**/*.wav')
sorted_audio_list.sort()

sorted_ground_truth_list = glob.glob('/Users/Jaedicke/MAPS/**/ISOL/**/*.txt')
sorted_ground_truth_list.sort()


num_examples = count_examples(sorted_ground_truth_list)
print(num_examples)
semitone_filter = None
semitone_freq_bins = None
layered_spectrogram = None
spectrogram_data = np.zeros((num_examples, 88*15*3))
ground_truth_data = np.zeros((num_examples, 88))
data_index = 0

for file_index in range(0, len(sorted_audio_list)):
#if(True):
    #file_index = 0
    # load ground truth
    ground_truth = loadtxt(sorted_ground_truth_list[file_index], skiprows=1, delimiter='\t')

    # load signal
    sig = madmom.audio.signal.Signal(sorted_audio_list[file_index], sample_rate=sr, num_channels=1, norm=True,
                                     dtype=np.float32)

    # split signal into frames
    framed_sig_512 = madmom.audio.signal.FramedSignal(sig, frame_size=frame_length_samples[0],
                                                      hop_size=hop_size, origin='right')
    framed_sig_1024 = madmom.audio.signal.FramedSignal(sig, frame_size=frame_length_samples[1],
                                                       hop_size=hop_size, origin='right')
    framed_sig_2048 = madmom.audio.signal.FramedSignal(sig, frame_size=frame_length_samples[2],
                                                       hop_size=hop_size, origin='right')

    # apply STFT
    stft_sig_512 = madmom.audio.stft.STFT(frames=framed_sig_512, fft_size=10000, circular_shift=True)
    stft_sig_1024 = madmom.audio.stft.STFT(frames=framed_sig_512, fft_size=10000, circular_shift=True)
    stft_sig_2048 = madmom.audio.stft.STFT(frames=framed_sig_512, fft_size=10000, circular_shift=True)

    # create filterbank
    if file_index == 0:
        semitone_filter, semitone_freq_bins = create_semitone_filterbank(stft_sig_512.bin_frequencies)
    # mel_filterbank, mel_freq_bins = create_mel_filterbank(stft_sig.bin_frequencies)

    # transform to spectrogram
    spec_sig_512 = madmom.audio.spectrogram.Spectrogram(stft_sig_512)
    spec_sig_1024 = madmom.audio.spectrogram.Spectrogram(stft_sig_1024)
    spec_sig_2048 = madmom.audio.spectrogram.Spectrogram(stft_sig_2048)

    # apply filter
    semitone_filt_sig_512 = np.dot(spec_sig_512, semitone_filter)
    semitone_filt_sig_1024 = np.dot(spec_sig_1024, semitone_filter)
    semitone_filt_sig_2048 = np.dot(spec_sig_2048, semitone_filter)
    # mel_filt_sig = np.dot(spec_sig, mel_filterbank)

    # apply log magnitude
    log_semitone_512 = madmom.audio.spectrogram.LogarithmicSpectrogram(semitone_filt_sig_512)
    log_semitone_1024 = madmom.audio.spectrogram.LogarithmicSpectrogram(semitone_filt_sig_1024)
    log_semitone_2048 = madmom.audio.spectrogram.LogarithmicSpectrogram(semitone_filt_sig_2048)
    # log_mel = madmom.audio.spectrogram.LogarithmicSpectrogram(mel_filt_sig)

    # check how long the list of ground truth's is
    ground_truth_length = 0
    if ground_truth.ndim == 1:
        ground_truth_length = 1
    else:
        ground_truth_length = ground_truth.shape[0]
    # iterate over all onsets in current file
    for ground_truth_index in range(0, ground_truth_length):

        # load onset at current ground truth index
        if ground_truth_length == 1:
            onset_seconds = ground_truth[0]
            ground_truth_midi = int(ground_truth[2])
        else:
            onset_seconds = ground_truth[ground_truth_index, 0]
            ground_truth_midi = int(ground_truth[ground_truth_index, 2])
        if True:
        #if ground_truth_midi >= 48 and ground_truth_midi <= 59:
            # three layer spectrogram placeholder
            layered_spectrogram = np.zeros((88, 15, 3), np.float32)

            # iterate over all three frame length settings

            # use only frame length of 512 samples since it provides the best time resolution to find the true frame with onset
            # with longer frame length the onset might be found in several frames
            center_frame = find_onset_frame(onset_seconds, frame_length_samples[0], hop_size, sr)
            # flip spectrograms up to down for convenient representation of filters in tensorboard. Filters are presented as
            # png files therefor the upper left corner is (0,0)
            layered_spectrogram[:, :, 0] = np.flipud(log_semitone_512[(center_frame-7):(center_frame+8), :].T)
            layered_spectrogram[:, :, 1] = np.flipud(log_semitone_1024[(center_frame-7):(center_frame+8), :].T)
            layered_spectrogram[:, :, 2] = np.flipud(log_semitone_2048[(center_frame-7):(center_frame+8), :].T)
            # flatten spectrogram and save to data matrix
            spectrogram_data[data_index, :] = layered_spectrogram.ravel()
            # save ground truth in binary array format
            ground_truth_data[data_index, ground_truth_midi-21] = 1
            #
            data_index = data_index + 1
            print(data_index)

#np.savez("single_note_56496_examples", features=spectrogram_data[0:data_index, :], labels=ground_truth_data[0:data_index, :])
#np.savez("semitone_single_note_56496_examples", features=spectrogram_data, labels=ground_truth_data)

# plot specs
#specshow(spec=log_semitone, title='Semitone Spectrogram', center_frame=center_frame, freq_bins=semitone_freq_bins)

#specshow(spec=log_mel, title='Mel Spectrogram', center_frame=center_frame, freq_bins=mel_freq_bins)


#np.savetxt('test.csv', log_semitone[(center_frame-7):(center_frame+8), :].T, delimiter=',')

#np.save(outfile, log_semitone[(center_frame-7):(center_frame+8), :].T)
#np.save(outfile, log_mel[(center_frame-7):(center_frame+8), :].T)

#outfile.seek(0)

#print(np.load(outfile).shape)
"""
data = np.load("single_note_56496_examples.npz")
print(data.files)
#print(data["features"])
print(data["labels"].shape)
sess = tf.InteractiveSession()
flat_tf = tf.convert_to_tensor(data["features"][data_index-1])
b = tf.reshape(flat_tf, [88, 15, 3])

# We can just use 'c.eval()' without passing 'sess'
#print(b.eval())
plt.figure()
plt.imshow(b.eval())
plt.show()
sess.close()
"""
#img = np.reshape(layered_spectrogram,(88,15,3))







