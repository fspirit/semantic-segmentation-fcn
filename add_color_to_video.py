import tensorflow as tf
import scipy
import numpy as np

import helper

from moviepy.editor import VideoFileClip

IMAGE_SHAPE = (160, 576)

def read_saved_model(sess):
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], './model/')

    graph = tf.get_default_graph()
    input_ph = graph.get_tensor_by_name('image_input:0')
    keep_prob_ph = graph.get_tensor_by_name('keep_prob:0')
    logits = graph.get_tensor_by_name('Reshape_2:0')

    return input_ph, keep_prob_ph, logits

def test_with_image():
    test_image_file = './data/data_road/testing/image_2/um_000001.png'
    image = scipy.misc.imresize(scipy.misc.imread(test_image_file), IMAGE_SHAPE)

    with tf.Session() as sess:
        input_ph, keep_prob_ph, logits = read_saved_model(sess)
        g_image = helper.get_painted_image(sess, logits, keep_prob_ph, input_ph, image, IMAGE_SHAPE)
        scipy.misc.imsave('./test_image.png', g_image)


def process_video(original_video_path):

    def process_frame(sess, logits, keep_prob_ph, input_ph, frame):
        resized_frame = scipy.misc.imresize(frame, IMAGE_SHAPE)
        g_image = helper.get_painted_image(sess, logits, keep_prob_ph, input_ph, resized_frame, IMAGE_SHAPE)

        return scipy.misc.imresize(g_image, IMAGE_SHAPE)

    original_video = VideoFileClip(original_video_path)
    with tf.Session() as sess:
        input_ph, keep_prob_ph, logits = read_saved_model(sess)

        output_video = original_video.fl_image(lambda frame: process_frame(sess, logits, keep_prob_ph, input_ph, frame))
        output_video.write_videofile('./video/output.mp4', audio=False)

if __name__ == '__main__':
    # import imageio
    # imageio.plugins.ffmpeg.download()
    process_video('./video/project_video_test_01.mp4')


