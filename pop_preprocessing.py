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
import tensorflow as tf


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


def create_semitone_filterbank(signal_file, sr, frame_length_samples, hop_size, bands_per_octave):
    # load signal
    sig = madmom.audio.signal.Signal(signal_file, sample_rate=sr, num_channels=1, norm=True,
                                     dtype=np.float32)

    # split signal into frames
    framed_sig_1, framed_sig_2, framed_sig_3 = framed_signal(sig, frame_size=frame_length_samples, hop_size=hop_size,
                                                               origin='right')
    # apply STFT
    stft_sig_1, stft_sig_2, stft_sig_3 = framed_signal_stft(framed_sig_1, framed_sig_2, framed_sig_3, 10000, True)

    frequencies = stft_sig_1.bin_frequencies

    semitone_freq_bins = madmom.audio.filters.log_frequencies(bands_per_octave=bands_per_octave, fmin=27.5,
                                                              fmax=4186.0090448096, fref=440.0)
    bins = madmom.audio.filters.frequencies2bins(frequencies, semitone_freq_bins, unique_bins=False)
    semitone_filterbank = np.zeros(shape=[88, len(bins)])
    current_bin = bins[0]
    start = 0
    for i in range(0, len(bins)):
        if bins[i] > current_bin:
            center = start + int(round((i-1 - start)/2))
            semitone_filterbank[current_bin][start:i] = madmom.audio.filters.TriangularFilter(start, center, i,
                                                                                              norm=True)
            current_bin = bins[i]
            start = i
    # last semitone triangle filter
    center = start + int(round((len(bins) - 1 - start) / 2))
    semitone_filterbank[current_bin][start:len(bins)-1] = madmom.audio.filters.TriangularFilter(start, center,
                                                                                                len(bins) - 1,
                                                                                                norm=True)

    return semitone_filterbank.T, semitone_freq_bins

def create_logarithmic_filterbank(sr, longest_frame, bands_per_octave):
    bin_frequencies = madmom.audio.stft.fft_frequencies(longest_frame, sr)
    log_frequencies = madmom.audio.filters.log_frequencies(bands_per_octave=bands_per_octave, fmin=27.5,
                                                           fmax=4186.0090448096, fref=440.0)
    log_filter = madmom.audio.filters.LogarithmicFilterbank(bin_frequencies, num_bands=bands_per_octave, fmin=27.5,
                                                            fmax=4186.0090448096, fref=440.0, norm_filters=True,
                                                            unique_filters=True, bands_per_octave=True)
    return log_filter, log_frequencies

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


def write_to_tfrecords(audio_list, label_list, filename, num_bands, num_frames, sr=22050):
    hop_size = librosa.time_to_samples(0.01, sr=sr)
    frame_length_samples = [512, 1024, 2048]

    num_examples = count_examples(label_list) * 5
    print("Total of " + str(num_examples) + " examples")

    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(str(num_examples) + "_" + filename + ".tfrecords")

    midi_note_offset = 21

    # offset from the center frame depending on the number of frames used
    left_offset_frame = np.floor(num_frames/2.0).astype(int)
    right_offset_frame = np.ceil(num_frames/2.0).astype(int)

    # weight factors
    weight_factors = [0.0, 0.25, 1.0, 0.25, 0.0]

    #spec_filter, semitone_freq_bins = create_semitone_filterbank(audio_list[0], sr=sr,
    #                                                             frame_length_samples=frame_length_samples,
    #                                                             hop_size=hop_size)

    spec_filter, log_frequencies = create_logarithmic_filterbank(sr=sr, longest_frame=frame_length_samples[2],
                                                                 bands_per_octave=num_bands)
    num_bins = np.shape(spec_filter)[1]

    num_classes = 88

    examples_processed = 0

    for file_index in range(0, len(audio_list)):
        # load signal
        sig = madmom.audio.signal.Signal(audio_list[file_index], sample_rate=sr, num_channels=1, norm=True,
                                         dtype=np.float32)

        # split signal into frames
        framed_sig_1, framed_sig_2, framed_sig_3 = framed_signal(sig, frame_length_samples, hop_size=hop_size,
                                                                 origin='right')

        # apply STFT (fft_size=10000 for semitone_filterbank
        stft_sig_1, stft_sig_2, stft_sig_3 = framed_signal_stft(framed_sig_1, framed_sig_2, framed_sig_3,
                                                                fft_size=frame_length_samples[2]*2, circular_shift=True)

        # transform to spectrogram
        spec_sig_1, spec_sig_2, spec_sig_3 = framed_stft_spectrogram(stft_sig_1, stft_sig_2, stft_sig_3)

        # apply filter
        filt_sig_1, filt_sig_2, filt_sig_3 = framed_filter_spectrogram(spec_sig_1, spec_sig_2, spec_sig_3, spec_filter)

        # apply log magnitude
        log_filt_1, log_filt_2, log_filt_3 = framed_log_magnitude_spectrogram(filt_sig_1, filt_sig_2, filt_sig_3)

        # load ground truth
        labeled_intervals = loadtxt(label_list[file_index], skiprows=1, delimiter='\t')

        # in case only one note exist in this example
        if labeled_intervals.ndim == 1:
            labeled_intervals = labeled_intervals.reshape(1, 3)

        # find unique onsets
        unique_onsets = np.unique(labeled_intervals[:, 0])

        # iterate over all onsets in current file
        for onset in unique_onsets:

            onset_occurence = np.where(labeled_intervals[:, 0] == onset)
            onset_midi_notes = labeled_intervals[onset_occurence, 2]
            onset_notes_index = onset_midi_notes.ravel().astype(int) - midi_note_offset

            if examples_processed < num_examples:
                # if ground_truth_midi >= 48 and ground_truth_midi <= 59:


                # iterate over all three frame length settings

                # use only frame length of 512 samples since it provides the best time resolution to find the true frame with onset
                # with longer frame length the onset might be found in several frames
                center_frame = find_onset_frame(onset, frame_length_samples[0], hop_size, sr)

                # three layer spectrogram placeholder
                layered_spectrogram = np.zeros((num_bins, num_frames, 3), np.float32)

                for index in range(-2, 3):
                    
                    # get slices from framed spec for this center frame
                    example = example_slice_from_frames(layered_spectrogram, log_filt_1, log_filt_2, log_filt_3,
                                                        (center_frame + index), left_offset_frame, right_offset_frame,
                                                        num_classes, onset_notes_index, filename, weight_factors[index+2])

                    # Serialize to string and write on the file
                    writer.write(example.SerializeToString())

                    examples_processed = examples_processed + 1
                    if np.mod(examples_processed, 100) == 0:
                        print(str(examples_processed) + " of " + str(num_examples) + " examples processed")
    writer.close()


def example_slice_from_frames(layered_spectrogram, log_filt_1, log_filt_2, log_filt_3, center_frame, left_offset_frame, right_offset_frame, num_classes, onset_notes_index, filename, weight_factor):
    # flip spectrograms up to down for convenient representation of filters in tensorboard. Filters are presented as
    # png files therefor the upper left corner is (0,0)
    layered_spectrogram[:, :, 0] = np.flipud(
        log_filt_1[(center_frame - left_offset_frame):(center_frame + right_offset_frame), :].T)
    layered_spectrogram[:, :, 1] = np.flipud(
        log_filt_2[(center_frame - left_offset_frame):(center_frame + right_offset_frame), :].T)
    layered_spectrogram[:, :, 2] = np.flipud(
        log_filt_3[(center_frame - left_offset_frame):(center_frame + right_offset_frame), :].T)

    label = np.zeros(num_classes, dtype=float)
    label[onset_notes_index] = 1.0 * weight_factor

    # Create a feature
    feature = {filename + "/label": _float_feature(label),
               filename + "/spec": _float_feature(layered_spectrogram.ravel())}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example



def main():
    sr = 22050
    frame_length_samples = [512, 1024, 2048]
    frame_length_seconds = np.divide(frame_length_samples, sr)
    num_frames = 15
    hop_size = librosa.time_to_samples(0.01, sr=sr)
    example_num = 1
    frame_length_num = 0

    subset = 'StbgTGd2'

    sorted_audio_list = glob.glob('/Users/Jaedicke/MAPS/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht.wav')
    sorted_audio_list.sort()

    sorted_ground_truth_list = glob.glob('/Users/Jaedicke/MAPS/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht.txt')
    sorted_ground_truth_list.sort()

    num_examples = count_examples(sorted_ground_truth_list)
    print("Total of " + str(num_examples) + " examples")

    midi_note_offset = 21

    spec_filter, semitone_freq_bins = create_semitone_filterbank(sorted_audio_list[0], sr=sr,
                                                                 frame_length_samples=frame_length_samples,
                                                                 hop_size=hop_size)
    semitone_freq_bins = None
    layered_spectrogram = None
    spectrogram_data = np.zeros((num_examples, 88*15*3))
    ground_truth_data = np.zeros((num_examples, 88))
    examples_processed = 0

    for file_index in range(0, len(sorted_audio_list)):
        # load signal
        sig = madmom.audio.signal.Signal(sorted_audio_list[file_index], sample_rate=sr, num_channels=1, norm=True,
                                         dtype=np.float32)

        # split signal into frames
        framed_sig_1, framed_sig_2, framed_sig_3 = framed_signal(sig, frame_length_samples, hop_size=hop_size, origin='right')

        # apply STFT
        stft_sig_1, stft_sig_2, stft_sig_3 = framed_signal_stft(framed_sig_1, framed_sig_2, framed_sig_3,
                                                                fft_size=10000, circular_shift=True)

        # transform to spectrogram
        spec_sig_1, spec_sig_2, spec_sig_3 = framed_stft_spectrogram(stft_sig_1, stft_sig_2, stft_sig_3)

        # apply filter
        filt_sig_1, filt_sig_2, filt_sig_3 = framed_filter_spectrogram(spec_sig_1, spec_sig_2, spec_sig_3, spec_filter)

        # apply log magnitude
        log_filt_1, log_filt_2, log_filt_3 = framed_log_magnitude_spectrogram(filt_sig_1, filt_sig_2, filt_sig_3)

        # load ground truth
        labeled_intervals = loadtxt(sorted_ground_truth_list[file_index], skiprows=1, delimiter='\t')

        # in case only one note exist in this example
        if labeled_intervals.ndim == 1:
            labeled_intervals = labeled_intervals.reshape(1, 3)

        # find unique onsets
        unique_onsets = np.unique(labeled_intervals[:, 0])

        # iterate over all onsets in current file
        for onset in unique_onsets:

            onset_occurence = np.where(labeled_intervals[:, 0] == onset)
            onset_midi_notes = labeled_intervals[onset_occurence, 2]
            onset_notes_index = onset_midi_notes.ravel().astype(int) - midi_note_offset



            if examples_processed < num_examples:
            #if ground_truth_midi >= 48 and ground_truth_midi <= 59:
                # three layer spectrogram placeholder
                layered_spectrogram = np.zeros((88, 15, 3), np.float32)

                # iterate over all three frame length settings

                # use only frame length of 512 samples since it provides the best time resolution to find the true frame with onset
                # with longer frame length the onset might be found in several frames
                center_frame = find_onset_frame(onset, frame_length_samples[0], hop_size, sr)
                # flip spectrograms up to down for convenient representation of filters in tensorboard. Filters are presented as
                # png files therefor the upper left corner is (0,0)
                layered_spectrogram[:, :, 0] = np.flipud(log_filt_1[(center_frame-7):(center_frame+8), :].T)
                layered_spectrogram[:, :, 1] = np.flipud(log_filt_2[(center_frame-7):(center_frame+8), :].T)
                layered_spectrogram[:, :, 2] = np.flipud(log_filt_3[(center_frame-7):(center_frame+8), :].T)
                # flatten spectrogram and save to data matrix
                spectrogram_data[examples_processed, :] = layered_spectrogram.ravel()
                # save ground truth in binary array format
                ground_truth_data[examples_processed, onset_notes_index] = 1

                examples_processed = examples_processed + 1
                if np.mod(examples_processed, 100) == 0:
                    print(str(examples_processed) + " of " + str(num_examples) + " examples processed")

    np.savez("semitone_MAPS_MUS-alb_se3_AkPnBcht_onsets_" + str(num_examples) + "_examples", features=spectrogram_data,
             labels=ground_truth_data)
    specshow(spec=np.flipud(layered_spectrogram[:, :, 1]), title='Semitone Spectrogram')
    print(onset_notes_index+21)


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








