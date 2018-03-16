import os
import sys

BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.insert(0, BASE_DIR)

import click
import numpy as np
import tensorflow as tf

from utils import measure_model
from models.tensorflow_model import Model
from constants import TENSORFLOW_SAVES_DIR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(42)


class OptimizedModel:
    def __init__(self, saves_dir, input_node_name, output_node_name, graph_name):
        # load saved optimized graph
        frozen_save_path = os.path.join(saves_dir, graph_name)
        with tf.gfile.Open(frozen_save_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as new_graph:
            tf.import_graph_def(graph_def, name="prefix")

        self.input = new_graph.get_tensor_by_name('prefix/%s:0' % input_node_name)
        self.output = new_graph.get_tensor_by_name('prefix/%s:0' % output_node_name)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=new_graph, config=config)

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

    optimized_graph = os.path.join(saves_dir, 'optimized_graph.pb')
    print("Optimized graph size:", os.path.getsize(optimized_graph))
    quantized_graph = os.path.join(saves_dir, 'quantized_graph.pb')
    print("Quantized graph size:", os.path.getsize(quantized_graph))

    for batch_size in batches:
        print("Batch size: {}".format(batch_size))
        batch = np.random.random((batch_size, 224, 224, 3))

        tf.reset_default_graph()
        frozen_model = OptimizedModel(
            saves_dir,
            input_node_name=Model.input_node_name,
            output_node_name=Model.output_node_name,
            graph_name='optimized_graph.pb'
        )
        measure_model(frozen_model, "Optim. model", batch)
        frozen_model.sess.close()

        tf.reset_default_graph()
        frozen_model = OptimizedModel(
            saves_dir,
            input_node_name=Model.input_node_name,
            output_node_name=Model.output_node_name,
            graph_name='quantized_graph.pb'
        )
        measure_model(frozen_model, "Quantized model", batch)
        frozen_model.sess.close()


if __name__ == '__main__':
    main()
