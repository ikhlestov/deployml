import numpy as np
import tensorflow as tf


class Model:
    input_node_name = 'input'
    output_node_name = 'output'
    input_data_type = tf.float32

    def __init__(self):
        self.input = tf.placeholder(
            self.input_data_type,
            shape=(None, 224, 224, 3),
            name=self.input_node_name
        )
        self.kernel_count = 0
        self._build_net()
        self._initialize_session()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def get_kernel_name(self):
        self.kernel_count += 1
        return 'kernel_{}'.format(self.kernel_count)

    def __kernel(self, in_features, out_features):
        return tf.get_variable(name=self.get_kernel_name(),
                               shape=(3, 3, in_features, out_features))

    def _build_net(self):
        # some constants
        strides = (1, 1, 1, 1)
        padding = "VALID"
        x = self.input

        # first block
        x = tf.nn.conv2d(x, filter=self.__kernel(3, 6), strides=strides, padding=padding)
        x = tf.nn.relu(tf.layers.batch_normalization(x))
        x = tf.nn.conv2d(x, filter=self.__kernel(6, 9), strides=strides, padding=padding)
        x = tf.nn.relu(tf.layers.batch_normalization(x))
        x = tf.nn.pool(x, (2, 2), 'AVG', padding=padding, strides=(2, 2))

        # second block
        x = tf.nn.conv2d(x, filter=self.__kernel(9, 12), strides=strides, padding=padding)
        x = tf.nn.relu(tf.layers.batch_normalization(x))
        x = tf.nn.conv2d(x, filter=self.__kernel(12, 16), strides=strides, padding=padding)
        x = tf.nn.relu(tf.layers.batch_normalization(x))
        x = tf.nn.pool(x, (2, 2), 'AVG', padding=padding, strides=(2, 2))

        # third block
        x = tf.nn.conv2d(x, filter=self.__kernel(16, 32), strides=strides, padding=padding)
        x = tf.nn.relu(tf.layers.batch_normalization(x))
        x = tf.nn.conv2d(x, filter=self.__kernel(32, 64), strides=strides, padding=padding)
        x = tf.nn.relu(tf.layers.batch_normalization(x))
        x = tf.nn.pool(x, (2, 2), 'AVG', padding=padding, strides=(2, 2))

        # forth block
        x = tf.nn.conv2d(x, filter=self.__kernel(64, 128), strides=strides, padding=padding)
        x = tf.nn.relu(tf.layers.batch_normalization(x))
        x = tf.nn.conv2d(x, filter=self.__kernel(128, 256), strides=strides, padding=padding)
        # x = self.large_block(x, strides)
        x = tf.nn.relu(tf.layers.batch_normalization(x))

        # transition to classes
        x = tf.nn.pool(x, (20, 20), 'AVG', padding=padding, strides=(1, 1))
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 512)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.dense(x, 100)
        x = tf.nn.sigmoid(x, name=self.output_node_name)
        self.output = x

    def predict(self, inputs):
        feed_dict = {self.input: inputs}
        pred = self.sess.run(self.output, feed_dict=feed_dict)
        return pred

    def large_block(self, x, strides):
        x = tf.nn.conv2d(x, filter=self.__kernel(256, 512), strides=strides, padding='SAME')
        x = tf.nn.conv2d(x, filter=self.__kernel(512, 512), strides=strides, padding='SAME')
        x = tf.nn.conv2d(x, filter=self.__kernel(512, 512), strides=strides, padding='SAME')
        x = tf.nn.conv2d(x, filter=self.__kernel(512, 256), strides=strides, padding='SAME')
        return x


if __name__ == '__main__':
    sample_image = np.random.random((1, 224, 224, 3))
    model = Model()
    preds = model.predict(sample_image)
