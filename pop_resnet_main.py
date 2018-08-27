from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import pop_resnet
import pop_input_data as dataset
import os
from official.utils.logs import logger
import numpy as np

train_dataset_fp = "/Users/Jaedicke/Documents/MATLAB/spectrogramComputation/ISOL_SEMI_FILT_C4toB4_TRIPEL.csv"
eval_dataset_fp = "/Users/Jaedicke/Documents/MATLAB/spectrogramComputation/ISOL_SEMI_FILT_C4toB4_TRIPEL_EVAL.csv"
test_dataset_fp = "/Users/Jaedicke/Documents/MATLAB/spectrogramComputation/MUS_SEMI_FILT_C4toB4_TRIPEL_TEST.csv"
predict_dataset_fp = "/Users/Jaedicke/Documents/MATLAB/spectrogramComputation/ISOL_SEMI_FILT_DUMMY.csv"

train_dataset = "semitone_single_note_56496_examples.npz"

eval_dataset = "MAPS_MUS-alb_se3_AkPnBcht_25050.npz"

DEFAULT_DTYPE = tf.float32

TEST_ID = 1

num_examples = 56496
batch_size = 128
steps_per_epoch = int(round(num_examples/batch_size))
train_epochs = 5
total_train_steps = train_epochs * steps_per_epoch

run_params = {
    'batch_size': batch_size,
    'dtype': DEFAULT_DTYPE,
    'resnet_size': 34,
    'resnet_version': 2,
    'num_classes': 88,
    'weight_decay': 2e-4,
    'train_steps': total_train_steps, # 1000
    'eval_steps': 2000, # 2000
    'data_format': 'channels_last',
    'loss_scale': 128 if DEFAULT_DTYPE == tf.float16 else 1,
    'train_epochs': train_epochs
}



def main(argv):

    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_steps=50,  # Save checkpoints every 50 steps.
        keep_checkpoint_max=10,  # Retain the 10 most recent checkpoints.
    )
    classifier = tf.estimator.Estimator(
        model_fn=pop_resnet.resnet_model_fn,
        model_dir="/Users/Jaedicke/tensorflow/one_octave_resnet/model",
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

    # Train the Model.
    #classifier.train(
    #    input_fn=dataset.numpy_array_input_fn(train_dataset, batch_size=run_params['batch_size'],
    #                                          num_epochs=run_params['train_epochs'], shuffle=True), steps=run_params['train_steps'])

    # Evaluate the model.
    #eval_result = classifier.evaluate(input_fn=dataset.numpy_array_input_fn(eval_dataset, batch_size=1, num_epochs=1, shuffle=False),
    #                                  steps=run_params['eval_steps'])

    #benchmark_logger.log_evaluation_result(eval_result)

    #print('\nEval set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    predictions = classifier.predict(input_fn=dataset.numpy_array_input_fn(eval_dataset, batch_size=1, num_epochs=1,
                                                                                shuffle=False))

    #print(list(predictions)[0])
    props = np.zeros((run_params['num_classes'], run_params['eval_steps']))
    index = 0
    for p in predictions:
        props[:, index] = p['probabilities'][:]
        index = index + 1
    np.savez("props1", props=props)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
