import numpy as np
from numpy import loadtxt
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import librosa
import madmom
import glob
import warnings
warnings.filterwarnings("ignore")
from scipy.signal import butter, filtfilt
import tensorflow as tf


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# this function finds the frame were the onset is nearest to the middle of the frame
def find_onset_frame(onset_in_sec, frame_length, hop_size, sample_rate):
    frame_onset_in_samples = onset_in_sec * sample_rate - (frame_length / 2)
    onset_in_frame = frame_onset_in_samples / hop_size

    return int(round(onset_in_frame))

def specshow(spec, title):
    plt.figure()
    plt.pcolormesh(spec)
    locs, labels = plt.yticks()
    # set last y-tick to 88
    locs[-1] = 88
    # find labels in frequency bins
    #plt.yticks(locs, np.append(freq_bins.round(decimals=1)[::10], freq_bins.round(decimals=1)[-1]))
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()


def midi_to_hz(midi_num, fref=440.0):
    return np.float_power(2, ((midi_num-69)/12)) * fref


def create_semitone_filterbank(signal_file, sr, frame_length_samples, hop_size):
    # load signal
    sig = madmom.audio.signal.Signal(signal_file, sample_rate=sr, num_channels=1, norm=True,
                                     dtype=np.float32)

    # split signal into frames
    framed_sig_1, framed_sig_2, framed_sig_3 = framed_signal(sig, frame_size=frame_length_samples, hop_size=hop_size,
                                                               origin='right')
    # apply STFT
    stft_sig_1, stft_sig_2, stft_sig_3 = framed_signal_stft(framed_sig_1, framed_sig_2, framed_sig_3, 10000, True)

    frequencies = stft_sig_1.bin_frequencies

    semitone_freq_bins = madmom.audio.filters.semitone_frequencies(fmin=27.5, fmax=4186.0090448096, fref=440.0)
    bins = madmom.audio.filters.frequencies2bins(frequencies, semitone_freq_bins, unique_bins=False)
    semitone_filterbank = np.zeros(shape=[88, len(bins)])
    current_bin = bins[0]
    start = 0
    for i in range(0, len(bins)):
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
        list_per_file = loadtxt(example_list[example], skiprows=1, delimiter='\t')
        if list_per_file.ndim == 1:
            list_per_file = list_per_file.reshape(1, 3)
        unique_examples = np.unique(list_per_file[:, 0])
        count = count + len(unique_examples)
    return count


def framed_signal(sig, frame_size, hop_size, origin):
    # split signal into frames
    framed_sig_1 = madmom.audio.signal.FramedSignal(sig, frame_size=frame_size[0],
                                                    hop_size=hop_size, origin=origin)
    framed_sig_2 = madmom.audio.signal.FramedSignal(sig, frame_size=frame_size[1],
                                                    hop_size=hop_size, origin=origin)
    framed_sig_3 = madmom.audio.signal.FramedSignal(sig, frame_size=frame_size[2],
                                                    hop_size=hop_size, origin=origin)
    return framed_sig_1, framed_sig_2, framed_sig_3


def framed_signal_stft(framed_sig_1, framed_sig_2, framed_sig_3, fft_size, circular_shift):
    # apply STFT
    stft_sig_1 = madmom.audio.stft.STFT(frames=framed_sig_1, fft_size=fft_size, circular_shift=circular_shift)
    stft_sig_2 = madmom.audio.stft.STFT(frames=framed_sig_2, fft_size=fft_size, circular_shift=circular_shift)
    stft_sig_3 = madmom.audio.stft.STFT(frames=framed_sig_3, fft_size=fft_size, circular_shift=circular_shift)
    return stft_sig_1, stft_sig_2, stft_sig_3


def framed_stft_spectrogram(stft_sig_1, stft_sig_2, stft_sig_3):
    spec_sig_1 = madmom.audio.spectrogram.Spectrogram(stft_sig_1)
    spec_sig_2 = madmom.audio.spectrogram.Spectrogram(stft_sig_2)
    spec_sig_3 = madmom.audio.spectrogram.Spectrogram(stft_sig_3)
    return spec_sig_1, spec_sig_2, spec_sig_3


def framed_filter_spectrogram(spec_sig_1, spec_sig_2, spec_sig_3, spec_filter):
    filt_sig_1 = np.dot(spec_sig_1, spec_filter)
    filt_sig_2 = np.dot(spec_sig_2, spec_filter)
    filt_sig_3 = np.dot(spec_sig_3, spec_filter)
    return filt_sig_1, filt_sig_2, filt_sig_3


def framed_log_magnitude_spectrogram(filt_sig_1, filt_sig_2, filt_sig_3):
    log_filt_1 = madmom.audio.spectrogram.LogarithmicSpectrogram(filt_sig_1)
    log_filt_2 = madmom.audio.spectrogram.LogarithmicSpectrogram(filt_sig_2)
    log_filt_3 = madmom.audio.spectrogram.LogarithmicSpectrogram(filt_sig_3)
    return log_filt_1, log_filt_2, log_filt_3


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_tfrecords(audio_list, label_list, filename, sr=22050):
    hop_size = librosa.time_to_samples(0.01, sr=sr)
    frame_length_samples = [512, 1024, 2048]

    num_positive_examples = count_examples(label_list)
    print("Total of " + str(num_positive_examples) + " positive examples")

    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(str(num_positive_examples) + "_" + filename + ".tfrecords")


    spec_filter, semitone_freq_bins = create_semitone_filterbank(audio_list[0], sr=sr,
                                                                 frame_length_samples=frame_length_samples,
                                                                 hop_size=hop_size)
    examples_processed = 0

    for file_index in range(0, len(audio_list)):
        # load signal
        sig = madmom.audio.signal.Signal(audio_list[file_index], sample_rate=sr, num_channels=1, norm=True,
                                         dtype=np.float32)

        # split signal into frames
        framed_sig_1, framed_sig_2, framed_sig_3 = framed_signal(sig, frame_length_samples, hop_size=hop_size,
                                                                 origin='right')
        num_examples = framed_sig_1.shape

        # apply STFT
        stft_sig_1, stft_sig_2, stft_sig_3 = framed_signal_stft(framed_sig_1, framed_sig_2, framed_sig_3,
                                                                fft_size=10000, circular_shift=True)

        # transform to spectrogram
        spec_sig_1, spec_sig_2, spec_sig_3 = framed_stft_spectrogram(stft_sig_1, stft_sig_2, stft_sig_3)

        # apply filter
        filt_sig_1, filt_sig_2, filt_sig_3 = framed_filter_spectrogram(spec_sig_1, spec_sig_2, spec_sig_3, spec_filter)

        # apply log magnitude
        log_filt_1, log_filt_2, log_filt_3 = framed_log_magnitude_spectrogram(filt_sig_1, filt_sig_2, filt_sig_3)

        for center_frame in range(7, num_examples[0]-8):
            if examples_processed < num_examples[0]:
                # if ground_truth_midi >= 48 and ground_truth_midi <= 59:
                # three layer spectrogram placeholder
                layered_spectrogram = np.zeros((88, 15, 3), np.float32)

                # iterate over all three frame length settings
                # flip spectrograms up to down for convenient representation of filters in tensorboard. Filters are presented as
                # png files therefor the upper left corner is (0,0)
                layered_spectrogram[:, :, 0] = np.flipud(log_filt_1[(center_frame - 7):(center_frame + 8), :].T)
                layered_spectrogram[:, :, 1] = np.flipud(log_filt_2[(center_frame - 7):(center_frame + 8), :].T)
                layered_spectrogram[:, :, 2] = np.flipud(log_filt_3[(center_frame - 7):(center_frame + 8), :].T)

                label = np.zeros(88, dtype=int)

                # Create a feature
                feature = {filename+"/label": _int64_feature(label),
                           filename+"/spec": _float_feature(layered_spectrogram.ravel())}
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

                examples_processed = examples_processed + 1
                if np.mod(examples_processed, 100) == 0:
                    print(str(examples_processed) + " of " + str(num_examples[0]) + " examples processed")
    writer.close()


def main():
    sr = 22050
    frame_length_samples = [512, 1024, 2048]
    frame_length_seconds = np.divide(frame_length_samples, sr)
    hop_size = librosa.time_to_samples(0.01, sr=sr)
    midi_low = 48
    midi_high = 59

    subset = 'StbgTGd2'

    sorted_audio_list = glob.glob('/Users/Jaedicke/MAPS/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht.wav')
    sorted_audio_list.sort()


    num_examples = 0


    midi_note_offset = 21

    spec_filter, semitone_freq_bins = create_semitone_filterbank(sorted_audio_list[0], sr=sr,
                                                                 frame_length_samples=frame_length_samples,
                                                                 hop_size=hop_size)
    semitone_freq_bins = None
    layered_spectrogram = None


    examples_processed = 0

    for file_index in range(0, len(sorted_audio_list)):
        # load signal
        sig = madmom.audio.signal.Signal(sorted_audio_list[file_index], sample_rate=sr, num_channels=1, norm=True,
                                         dtype=np.float32)

        sig = butter_bandpass_filter(sig, midi_to_hz(midi_low), midi_to_hz(midi_high), sr, order=2)

        # split signal into frames
        framed_sig_1, framed_sig_2, framed_sig_3 = framed_signal(sig, frame_length_samples, hop_size=hop_size, origin='right')
        num_examples = framed_sig_1.shape

        print("Total of " + str(num_examples[0]-7-8) + " examples")
        spectrogram_data = np.zeros((num_examples[0]-7-8, 88 * 15 * 3))
        ground_truth_data = np.zeros((num_examples[0]-7-8, 88))

        # apply STFT
        stft_sig_1, stft_sig_2, stft_sig_3 = framed_signal_stft(framed_sig_1, framed_sig_2, framed_sig_3,
                                                                fft_size=10000, circular_shift=True)

        # transform to spectrogram
        spec_sig_1, spec_sig_2, spec_sig_3 = framed_stft_spectrogram(stft_sig_1, stft_sig_2, stft_sig_3)

        # apply filter
        filt_sig_1, filt_sig_2, filt_sig_3 = framed_filter_spectrogram(spec_sig_1, spec_sig_2, spec_sig_3, spec_filter)

        # apply log magnitude
        log_filt_1, log_filt_2, log_filt_3 = framed_log_magnitude_spectrogram(filt_sig_1, filt_sig_2, filt_sig_3)

        for center_frame in range(7, num_examples[0]-8):
            if examples_processed < num_examples[0]:
                # three layer spectrogram placeholder
                layered_spectrogram = np.zeros((88, 15, 3), np.float32)

                # iterate over all three frame length settings

                # use only frame length of 512 samples since it provides the best time resolution to find the true frame with onset
                # with longer frame length the onset might be found in several frames

                # flip spectrograms up to down for convenient representation of filters in tensorboard. Filters are presented as
                # png files therefor the upper left corner is (0,0)
                layered_spectrogram[:, :, 0] = np.flipud(log_filt_1[(center_frame-7):(center_frame+8), :].T)
                layered_spectrogram[:, :, 1] = np.flipud(log_filt_2[(center_frame-7):(center_frame+8), :].T)
                layered_spectrogram[:, :, 2] = np.flipud(log_filt_3[(center_frame-7):(center_frame+8), :].T)
                # flatten spectrogram and save to data matrix
                spectrogram_data[examples_processed, :] = layered_spectrogram.ravel()

                examples_processed = examples_processed + 1
                if np.mod(examples_processed, 100) == 0:
                    print(str(examples_processed) + " of " + str(num_examples[0]-7-8) + " examples processed")

    np.savez("semitone_MAPS_MUS-alb_se3_AkPnBcht_" + str(examples_processed) + "_examples", features=spectrogram_data,
             labels=ground_truth_data)
    #specshow(spec=np.flipud(layered_spectrogram[:, :, 1]), title='Semitone Spectrogram')



def load_preprocessed_data(filepath):
    data = np.load(filepath)
    features = data["features"]
    labels = data["labels"]
    return features, labels


def append_preprocessed_data(files, filename):
    features, labels = load_preprocessed_data(files[0])
    for filepath_index in range(1, len(files)):
        temp_features, temp_labels = load_preprocessed_data(files[filepath_index])
        features = np.append(features, temp_features, axis=0)
        labels = np.append(labels, temp_labels, axis=0)
    np.savez(filename, features=features, labels=labels)

def save_note_range(filepath, new_file, midi_range):
    features, labels = load_preprocessed_data(filepath=filepath)
    array_range = midi_range - 21
    total_notes_in_example = np.sum(labels, axis=1)
    notes_in_range = np.sum(labels[:, array_range[0]:array_range[1]+1], axis=1)
    range_indices = np.where(notes_in_range == total_notes_in_example)
    features_in_range = np.squeeze(features[range_indices, :])
    labels_in_range = np.squeeze(labels[range_indices, :])
    print(labels_in_range.shape)
    np.savez(new_file + "_" + str(np.size(labels_in_range, axis=0)) + "_examples", features=features_in_range, labels=labels_in_range)




#np.savez("single_note_56496_examples", features=spectrogram_data[0:data_index, :], labels=ground_truth_data[0:data_index, :])



# plot specs


#specshow(spec=log_mel, title='Mel Spectrogram', center_frame=center_frame, freq_bins=mel_freq_bins)


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








