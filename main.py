import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

from model import FCNModel

# Check TensorFlow Version


assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

NUM_CLASSES = 2
IMAGE_SHAPE = (160, 576)
DATA_DIR = './data'
RUNS_DIR = './runs'
EPOCHS = 1
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
DROPOUT_KEEP_PROB = 0.75

def run():
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        vgg_path = os.path.join(DATA_DIR, 'vgg')
        get_batches_fn_train, get_batches_fn_val = helper.gen_batch_functions(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        model = FCNModel(IMAGE_SHAPE, NUM_CLASSES, vgg_path)

        model.init(sess)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        model.train(sess, EPOCHS, BATCH_SIZE, get_batches_fn_train, DROPOUT_KEEP_PROB, LEARNING_RATE)

        model.predict(sess, BATCH_SIZE, get_batches_fn_val)

        # Run the model with the test images and save each painted output image (roads painted green)
        # helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE, model.logits, model.keep_prob_tf, model.input_tf)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    # run_tests()
    run()
