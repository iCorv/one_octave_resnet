from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import configurations.pop_hyper_parameters as php
import pop_input_data as dataset
import os
from official.utils.logs import logger
import numpy as np
import pop_model
import glob

train_dataset_tfrecord = glob.glob("./tfrecords-dataset/sigtia-configuration2-splits/fold_1/train/*.tfrecords")
val_dataset_tfrecord = glob.glob("./tfrecords-dataset/sigtia-configuration2-splits/fold_1/valid/*.tfrecords")
test_dataset_tfrecord = glob.glob("./tfrecords-dataset/sigtia-configuration2-splits/fold_1/test/*.tfrecords")

DEFAULT_DTYPE = tf.float32

TEST_ID = 1

train_and_val = True
predict_flag = False
train_flag = False
eval_flag = False

hparams = php.get_hyper_parameters('ResNet_v1')


def main(_):

    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_secs=300,  # Save checkpoints every 50 steps.
        keep_checkpoint_max=50,  # Retain the 10 most recent checkpoints.
        log_step_count_steps=1000
    )
    classifier = tf.estimator.Estimator(
        model_fn=pop_model.conv_net_model_fn,
        # model_dir="/Users/Jaedicke/tensorflow/one_octave_resnet/model",
        # model_dir="/Users/Jaedicke/tensorflow/model/model",
        model_dir="./model",
        # model_dir="D:/Users/cjaedicke/one_octave_resnet/model",
        config=estimator_config,
        params=hparams)

    benchmark_logger = logger.get_benchmark_logger()
    #benchmark_logger.log_run_info('ConvNet', 'MAPS', hparams, test_id=TEST_ID)

    # Train and validate in turns
    if train_and_val:
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: dataset.tfrecord_train_input_fn(train_dataset_tfrecord,
                                                             batch_size=hparams['batch_size'],
                                                             num_epochs=hparams['train_epochs']),
            max_steps=hparams['train_steps'])
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: dataset.tfrecord_val_input_fn(val_dataset_tfrecord,
                                                           batch_size=hparams['batch_size'],
                                                           num_epochs=1),
            steps=hparams['eval_steps'], throttle_secs=300)
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    # Train the Model.
    if train_flag:
        classifier.train(
            input_fn=lambda: dataset.tfrecord_train_input_fn(train_dataset_tfrecord,
                                                             batch_size=hparams['batch_size'],
                                                             num_epochs=hparams['train_epochs']),
            steps=hparams['train_steps'])

    # Evaluate the model.
    if eval_flag:
        eval_result = classifier.evaluate(
            input_fn=lambda: dataset.tfrecord_val_input_fn(test_dataset_tfrecord,
                                                           batch_size=hparams['batch_size'],
                                                           num_epochs=1),
            steps=hparams['test_steps'], checkpoint_path="./model/model.ckpt-1227258")
        benchmark_logger.log_evaluation_result(eval_result)

    # 1339892
    # Predict
    if predict_flag:
        predictions = classifier.predict(input_fn=lambda: dataset.tfrecord_test_input_fn(filepath=test_dataset_tfrecord,
                                                                                         batch_size=1, num_epochs=1), checkpoint_path="./model/model.ckpt-312775")

        # Problem: due to graph structure the value needs to be determined at compilation time?!
        num_test_frames = 28442
        # pythonic way to count elements in generator object
        # num_test_frames = len(list(predictions)) #sum(1 for i in predictions)
        print(num_test_frames)
        props = np.zeros((hparams['num_classes'], num_test_frames))
        notes = np.zeros((hparams['num_classes'], num_test_frames))
        index = 0
        for p in predictions:
            if index < num_test_frames:  #hparams['num_val_examples']:
                # print(np.shape(p['probabilities'][:]))
                props[:, index] = p['probabilities'][:]
                notes[:, index] = p['classes'][:]
            index = index + 1
        np.savez("props_MAPS_MUS-bor_ps6_ENSTDkCl_2018-11-11", props=props)
        # np.savez("notes_MAPS_MUS-chpn_op7_1_ENSTDkAm_2018-18-10", notes=notes)
        print(index)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
