from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import pop_resnet
import pop_input_data as dataset
import os
from official.utils.logs import logger

train_dataset_fp = "/Users/Jaedicke/Documents/MATLAB/spectrogramComputation/ISOL_SEMI_FILT_C4toB4_TRIPEL.csv"
eval_dataset_fp = "/Users/Jaedicke/Documents/MATLAB/spectrogramComputation/ISOL_SEMI_FILT_C4toB4_TRIPEL_EVAL.csv"
test_dataset_fp = "/Users/Jaedicke/Documents/MATLAB/spectrogramComputation/MUS_SEMI_FILT_C4toB4_TRIPEL_TEST.csv"
predict_dataset_fp = "/Users/Jaedicke/Documents/MATLAB/spectrogramComputation/ISOL_SEMI_FILT_DUMMY.csv"

DEFAULT_DTYPE = tf.float32

TEST_ID = 1

run_params = {
    'batch_size': 50,
    'dtype': DEFAULT_DTYPE,
    'resnet_size': 34,
    'resnet_version': 2,
    'num_classes': 12,
    'weight_decay': 2e-4,
    'train_steps': 1000,
    'eval_steps': 2000,
    'data_format': 'channels_last',
    'loss_scale': 128 if DEFAULT_DTYPE == tf.float16 else 1,
    'train_epochs': 10
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
    classifier.train(
        input_fn=lambda: dataset.csv_input_fn(train_dataset_fp, batch_size=run_params['batch_size']),
        steps=run_params['train_steps'])

    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda: dataset.csv_input_fn(test_dataset_fp, batch_size=1),
                                      steps=run_params['eval_steps'])

    benchmark_logger.log_evaluation_result(eval_result)

    print('\nEval set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


    #predictions = classifier.predict(
    #    input_fn=lambda: dataset.csv_input_fn(eval_dataset_fp, batch_size=1))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
