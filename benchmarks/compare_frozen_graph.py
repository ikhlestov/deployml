import os
import sys

BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.insert(0, BASE_DIR)

import click
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2

from utils import measure_model
from models.tensorflow_model import Model
from constants import TENSORFLOW_SAVES_DIR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(42)


class FrozenModel:
    def __init__(self, saves_dir):
        serialized_path = os.path.join(saves_dir, 'constant_graph.pb')
        input_graph_def = graph_pb2.GraphDef()
        with tf.gfile.Open(serialized_path, "rb") as f:
            input_graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(input_graph_def, name="prefix")

            self.input = graph.get_tensor_by_name('prefix/input:0')
            self.output = graph.get_tensor_by_name('prefix/output:0')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)

    def predict(self, inputs):
        feed_dict = {self.input: inputs}
        pred = self.sess.run(self.output, feed_dict=feed_dict)
        return pred


@click.command()
@click.option('-b', '--batch_size', type=int,
              help="Batch size for the sample input.")
def main(batch_size, saves_dir=TENSORFLOW_SAVES_DIR):
    saves_dir = os.path.join(BASE_DIR, saves_dir)
    batches = [1, 8, 16, 32, 64]
    if batch_size:
        batches = [batch_size]

    for batch_size in batches:
        print("Batch size: {}".format(batch_size))
        batch = np.random.random((batch_size, 224, 224, 3))

        tf.reset_default_graph()
        usual_model = Model()
        measure_model(usual_model, "Usual model", batch)
        usual_model.sess.close()

        tf.reset_default_graph()
        frozen_model = FrozenModel(saves_dir)
        measure_model(frozen_model, "Frozen model", batch)
        frozen_model.sess.close()


if __name__ == '__main__':
    main()
