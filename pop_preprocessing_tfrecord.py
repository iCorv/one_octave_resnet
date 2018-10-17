from random import shuffle
import glob
import pop_preprocessing as prep
shuffle_data = True  # shuffle the addresses before saving

# configuration II (using real piano recording as test set)
num_train_pieces = 180
num_val_pieces = 30
num_test_pieces = 60

# data folder
train_val_path_audio = '/Users/Jaedicke/MAPS/**/MUS/*.wav'
train_val_path_label = '/Users/Jaedicke/MAPS/**/MUS/*.txt'

test_path_audio = '/Users/Jaedicke/MAPS_real_piano/**/MUS/*.wav'
test_path_label = '/Users/Jaedicke/MAPS_real_piano/**/MUS/*.txt'

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

# number of musical pieces
num_files = len(audio_addrs)


# Divide the randomized train/validate data
train_audio_addrs = audio_addrs[0:num_train_pieces]
train_label_addrs = label_addrs[0:num_train_pieces]
val_audio_addrs = audio_addrs[num_train_pieces:]
val_label_addrs = label_addrs[num_train_pieces:]

# check number of pieces in each group
print("train/validate/test - split: " + str(len(train_audio_addrs)) + "/" + str(len(val_audio_addrs)) + "/" + str(len(test_audio_addrs)))

prep.write_to_tfrecords(train_audio_addrs, train_label_addrs, "train")
prep.write_to_tfrecords(val_audio_addrs, val_label_addrs, "val")
prep.write_to_tfrecords(test_audio_addrs, test_label_addrs, "test")
