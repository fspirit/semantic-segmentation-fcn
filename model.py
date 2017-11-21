import tensorflow as tf
import numpy as np

class FCNModel(object):

    def __init__(self, image_shape, num_classes, vgg_path):
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.vgg_path = vgg_path

        self.correct_label_tf = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        self.learning_rate_tf = tf.placeholder(tf.float32)

    def init(self, sess):
        layer_3, layer_4, layer_7 = self.load_vgg(sess)
        last_layer = self.build_layers(layer_3, layer_4, layer_7)
        self.build_train_and_loss_ops(last_layer)

    def load_vgg(self, sess):
        vgg_tag = 'vgg16'
        vgg_input_tensor_name = 'image_input:0'
        vgg_keep_prob_tensor_name = 'keep_prob:0'
        vgg_layer3_out_tensor_name = 'layer3_out:0'
        vgg_layer4_out_tensor_name = 'layer4_out:0'
        vgg_layer7_out_tensor_name = 'layer7_out:0'

        tf.saved_model.loader.load(sess, [vgg_tag], self.vgg_path)
        graph = tf.get_default_graph()

        self.input_tf = graph.get_tensor_by_name(vgg_input_tensor_name)
        self.keep_prob_tf = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

        layer_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

        return layer_3, layer_4, layer_7

    def build_train_and_loss_ops(self, last_layer):
        # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
        self.logits = tf.reshape(last_layer, (-1, self.num_classes))
        class_labels = tf.reshape(self.correct_label_tf, (-1, self.num_classes))

        # The cross_entropy_loss is the cost which we are trying to minimize to yield higher accuracy
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=class_labels)
        self.cross_entropy_loss = tf.reduce_mean(cross_entropy)

        # The model implements this operation to find the weights/parameters that would yield correct pixel labels
        self.train_op = tf.train.AdamOptimizer(self.learning_rate_tf).minimize(self.cross_entropy_loss)

    def build_layers(self, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out):
        # Apply a 1x1 convolution to encoder layers
        layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, self.num_classes, 1, padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, self.num_classes, 1, padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, self.num_classes, 1, padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

        output = tf.layers.conv2d_transpose(layer7_1x1, self.num_classes, 4, 2, padding='same',
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        output = tf.add(output, layer4_1x1)

        output = tf.layers.conv2d_transpose(output, self.num_classes, 4, 2, padding='same',
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        output = tf.add(output, layer3_1x1)
        output = tf.layers.conv2d_transpose(output, self.num_classes, 16, 8, padding='same',
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        return output

    def train(self, sess, epochs, batch_size, get_batches_fn, dropout_keep_prob, learning_rate):
        print("Training...")

        for epoch in range(epochs):
            print("Epoch ", epoch)
            batch_losses, i = [], 0

            for images, labels in get_batches_fn(batch_size):
                feed = {self.input_tf: images,
                        self.correct_label_tf: labels,
                        self.keep_prob_tf: dropout_keep_prob,
                        self.learning_rate_tf: learning_rate}

                _, batch_loss = sess.run([self.train_op, self.cross_entropy_loss], feed_dict=feed)

                print("Epoch ", epoch, "Iteration: ", i, " loss:", batch_loss)
                i += 1

    def predict(self, sess, batch_size, get_batches_fn_val):
        print("Validation ...")

        val_losses = []
        for val_images, val_labels in get_batches_fn_val(batch_size):
            feed = {self.input_tf: val_images,
                    self.correct_label_tf: val_labels,
                    self.keep_prob_tf: 1.0}

            val_batch_loss = sess.run(self.cross_entropy_loss, feed_dict=feed)
            val_losses.append(val_batch_loss)
        print("Validation loss ", np.mean(val_losses))


