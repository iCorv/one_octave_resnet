from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import pop_resnet
import pop_input_data as dataset
import os
from official.utils.logs import logger
from official.utils.logs import hooks_helper
import numpy as np
import pop_conv_net_kelz
import glob

tf.logging.set_verbosity(tf.logging.INFO)

train_dataset_tfrecord = glob.glob("./tfrecords-dataset/sigtia-configuration2-splits/fold_benchmark/train/*.tfrecords")
val_dataset_tfrecord = glob.glob("./tfrecords-dataset/sigtia-configuration2-splits/fold_benchmark/valid/*.tfrecords")
test_dataset_tfrecord = "./tfrecords-dataset/sigtia-configuration2-splits/fold_benchmark/test/MAPS_MUS-bor_ps6_ENSTDkCl.tfrecords"

DEFAULT_DTYPE = tf.float32

TEST_ID = 1

train_and_val = True
predict_flag = False
train_flag = False
eval_flag = False

num_examples = 1042876 #4163882
num_val_examples = 71435 #792567
batch_size = 128
steps_per_epoch = int(round(num_examples/batch_size))
train_epochs = 5
total_train_steps = train_epochs * steps_per_epoch

run_params = {
    'batch_size': batch_size,
    'dtype': DEFAULT_DTYPE,
    'resnet_size': 18,
    'resnet_version': 2,
    'num_classes': 88,
    'weight_decay': 2e-4,
    'train_steps': total_train_steps, # 1000
    'eval_steps': int(round(num_val_examples/batch_size)), # 1305, #25050, # 2000
    'data_format': 'channels_last',
    'loss_scale': 128 if DEFAULT_DTYPE == tf.float16 else 1,
    'train_epochs': train_epochs
}



def main(argv):

    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_steps=500,  # Save checkpoints every 50 steps.
        keep_checkpoint_max=2,  # Retain the 10 most recent checkpoints.
    )
    classifier = tf.estimator.Estimator(
        model_fn=pop_conv_net_kelz.conv_net_model_fn,
        #model_fn=pop_resnet.resnet_model_fn,
        #model_dir="/home/ubuntu/one_octave_resnet/model",
        model_dir="/Users/Jaedicke/tensorflow/one_octave_resnet/model",
        #model_dir="/Users/Jaedicke/tensorflow/model/model",
        #model_dir="D:/Users/cjaedicke/one_octave_resnet/model",
        config=estimator_config,
        params={'weight_decay': run_params['weight_decay'],
                'resnet_size': run_params['resnet_size'],
                'data_format': "channels_last",
                'batch_size': run_params['batch_size'],
                'resnet_version': run_params['resnet_version'],
                'loss_scale': run_params['loss_scale'],
                'dtype': run_params['dtype'],
                'num_classes': run_params['num_classes']
                })

    benchmark_logger = logger.get_benchmark_logger()
    benchmark_logger.log_run_info('resnet', 'MAPS', run_params,
                                  test_id=TEST_ID)

    #tensors_to_log = {"precision": "precision/value", "recall": "recall/value", "f1_score": "f1_score", "fn": "fn", "fp": "fp",
    #                  "tp": "tp"}
    #logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    #train_hooks = hooks_helper.get_logging_tensor_hook(every_n_iter=100, tensors_to_log=tensors_to_log)

    if train_and_val:
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: dataset.tfrecord_train_input_fn(train_dataset_tfrecord,
                                                                                             batch_size=run_params['batch_size'],
                                            num_epochs=run_params['train_epochs']), max_steps=total_train_steps)#, hooks=[train_hooks])
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: dataset.tfrecord_val_input_fn(val_dataset_tfrecord,
                                                                                         batch_size=run_params['batch_size'],
                                                                                         num_epochs=1),
                                          steps=run_params['eval_steps'], throttle_secs=300)

        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    # Train the Model.
    if train_flag:
        classifier.train(input_fn=lambda: dataset.tfrecord_train_input_fn(train_dataset_tfrecord,
                                                                          batch_size=run_params['batch_size'],
                                                                          num_epochs=run_params['train_epochs']),
                         steps=run_params['train_steps'])

    # Evaluate the model.
    if eval_flag:
        eval_result = classifier.evaluate(input_fn=lambda: dataset.tfrecord_val_input_fn(val_dataset_tfrecord,
                                                                                         batch_size=run_params['batch_size'],
                                                                                         num_epochs=1),
                                          steps=run_params['eval_steps'])

        benchmark_logger.log_evaluation_result(eval_result)

        print('\nEval set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    ######### predict
    if predict_flag:
        predictions = classifier.predict(input_fn=lambda: dataset.tfrecord_test_input_fn(filepath=test_dataset_tfrecord,
                                                                                         batch_size=1, num_epochs=1))

        # pythonic way to count elements in generator object?
        num_test_frames = 11468
        #num_test_frames = len(list(predictions)) #sum(1 for i in predictions)
        print(num_test_frames)
        props = np.zeros((run_params['num_classes'], num_test_frames))
        notes = np.zeros((run_params['num_classes'], num_test_frames))
        index = 0
        for p in predictions:
            if index < num_val_examples:
            #print(np.shape(p['probabilities'][:]))
                props[:, index] = p['probabilities'][:]
                notes[:, index] = p['classes'][:]
            index = index + 1
        np.savez("props_MAPS_MUS-bor_ps6_ENSTDkCl_2018-11-11", props=props)
        #np.savez("notes_MAPS_MUS-chpn_op7_1_ENSTDkAm_2018-18-10", notes=notes)
        print(index)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
