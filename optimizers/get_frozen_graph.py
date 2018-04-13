import os

import tensorflow as tf

from misc.constants import TENSORFLOW_SAVES_DIR
from models.tensorflow_model import Model


def main(saves_dir=TENSORFLOW_SAVES_DIR):
    # load previously saved model
    model = Model()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, os.path.join(saves_dir, 'usual_model'))

    # get constant graph
    constant_graph = tf.graph_util.convert_variables_to_constants(
        model.sess,
        tf.get_default_graph().as_graph_def(),
        [model.output.name.split(':')[0]]
    )

    # save constant graph
    frozen_save_path = os.path.join(saves_dir, 'constant_graph.pb')
    with tf.gfile.GFile(frozen_save_path, "wb") as f:
        f.write(constant_graph.SerializeToString())


if __name__ == '__main__':
    main()
