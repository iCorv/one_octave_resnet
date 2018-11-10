from random import shuffle
import glob
import pop_preprocessing as prep
import numpy as np
shuffle_data = True  # shuffle the addresses before saving

# number of frequency bands (per octave for logarithmic and total for mel-scaled)
num_bands = 48

# configuration II (using real piano recording as test set)
num_train_pieces = 180
num_val_pieces = 30
num_test_pieces = 60

# data folder
train_val_path_audio = '/Users/Jaedicke/MAPS/**/MUS/*.wav'
train_val_path_label = '/Users/Jaedicke/MAPS/**/MUS/*.txt'
#train_val_path_audio = 'D:/Users/cjaedicke/MAPS/**/MUS/*.wav'
#train_val_path_label = 'D:/Users/cjaedicke/MAPS/**/MUS/*.txt'

test_path_audio = '/Users/Jaedicke/MAPS_real_piano/**/MUS/*.wav'
test_path_label = '/Users/Jaedicke/MAPS_real_piano/**/MUS/*.txt'
#test_path_audio = 'D:/Users/cjaedicke/MAPS_real_piano/**/MUS/*.wav'
#test_path_label = 'D:/Users/cjaedicke/MAPS_real_piano/**/MUS/*.txt'

# read all file paths from the folders
train_val_audio_list = glob.glob(train_val_path_audio)
train_val_label_list = glob.glob(train_val_path_label)
test_audio_addrs = glob.glob(test_path_audio)
test_label_addrs = glob.glob(test_path_label)

# sort the list so we are sure a audio file and a label are at the same index
train_val_audio_list.sort()
train_val_label_list.sort()
test_audio_addrs.sort()
test_label_addrs.sort()

# shuffle data
if shuffle_data:
    c = list(zip(train_val_audio_list, train_val_label_list))
    shuffle(c)
    audio_addrs, label_addrs = zip(*c)

# save data order for reuse (e.g. creating other data folds)
np.savez("shuffeled_file_addrs", audio_addrs=audio_addrs, label_addrs=label_addrs)

#addrs = np.load("shuffeled_file_addrs.npz")
#audio_addrs=addrs['audio_addrs']
#label_addrs=addrs['label_addrs']

# number of musical pieces
num_files = len(audio_addrs)

#filenames = open(foldname, 'r').readlines()
#filenames = [f.strip() for f in filenames]

# Divide the randomized train/validate data
train_audio_addrs = audio_addrs[0:num_train_pieces]
train_label_addrs = label_addrs[0:num_train_pieces]
val_audio_addrs = audio_addrs[num_train_pieces:]
val_label_addrs = label_addrs[num_train_pieces:]

# check number of pieces in each group
print("train/validate/test - split: " + str(len(train_audio_addrs)) + "/" + str(len(val_audio_addrs)) + "/" + str(len(test_audio_addrs)))

#mean1, mean2, mean3 = prep.spec_mean_per_bin(val_audio_addrs, num_bands=num_bands)

#mean = np.load("mean_per_bin.npz")

#prep.spec_var_per_bin(val_audio_addrs, mean_1=mean['bin_mean_1'], mean_2=mean['bin_mean_2'], mean_3=mean['bin_mean_3'], filename='var_per_bin', num_bands=num_bands)


prep.write_piece_to_tfrecords(val_audio_addrs, val_label_addrs, "val", num_bands=num_bands, num_frames=5)
prep.write_piece_to_tfrecords(train_audio_addrs, train_label_addrs, "train", num_bands=num_bands, num_frames=5)

#prep.write_to_tfrecords(test_audio_addrs, test_label_addrs, "test")
