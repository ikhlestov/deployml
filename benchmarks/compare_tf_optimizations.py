import os

import click
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2

from misc.utils import measure_model
from misc.constants import TENSORFLOW_SAVES_DIR
from models.tensorflow_model import Model


class BinaryModel:
    def __init__(self, saves_dir, model_file, input_node_name, output_node_name):
        # read binary file
        binary_path = os.path.join(saves_dir, model_file)
        if not os.path.exists(binary_path):
            raise FileNotFoundError
        with tf.gfile.Open(binary_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # load readed content to the graph
        with tf.Graph().as_default() as new_graph:
            tf.import_graph_def(graph_def, name="prefix")

        # get input and output nodes
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
    """
    default model
    frozen model
    optimized frozen model
    quantized model
    """
    batches = [1, 8, 16, 32, 64]
    if batch_size:
        batches = [batch_size]

    for batch_size in batches:
        print("Batch size: {}".format(batch_size))
        batch = np.random.random((batch_size, 224, 224, 3))

        # our default model
        tf.reset_default_graph()
        usual_model = Model()
        measure_model(usual_model, "Usual model", batch)
        usual_model.sess.close()

        # our binary file
        tf.reset_default_graph()
        frozen_model = BinaryModel(
            saves_dir=saves_dir,
            model_file='constant_graph.pb',
            input_node_name=Model.input_node_name,
            output_node_name=Model.output_node_name
        )
        measure_model(frozen_model, "Frozen model", batch)
        frozen_model.sess.close()

        # binary file with some constant operations
        tf.reset_default_graph()
        optimized_frozen_model = BinaryModel(
            saves_dir=saves_dir,
            model_file='optimized_graph.pb',
            input_node_name=Model.input_node_name,
            output_node_name=Model.output_node_name
        )
        measure_model(optimized_frozen_model, "Optimized frozen model", batch)
        optimized_frozen_model.sess.close()

        # model quantized with python
        model_name = "Quantized with python"
        try:
            tf.reset_default_graph()
            optimized_frozen_model = BinaryModel(
                saves_dir=saves_dir,
                model_file='quantized_graph_python.pb',
                input_node_name=Model.input_node_name,
                output_node_name=Model.output_node_name
            )
            measure_model(optimized_frozen_model, model_name, batch)
            optimized_frozen_model.sess.close()
        except FileNotFoundError:
            print("skipped                                   // %s" % model_name)

        # model quantized with bazel
        model_name = "Quantized with bazel"
        try:
            tf.reset_default_graph()
            optimized_frozen_model = BinaryModel(
                saves_dir=saves_dir,
                model_file='quantized_graph_bazel.pb',
                input_node_name=Model.input_node_name,
                output_node_name=Model.output_node_name
            )
            measure_model(optimized_frozen_model, model_name, batch)
            optimized_frozen_model.sess.close()
        except FileNotFoundError:
            print("skipped                                   // %s" % model_name)


if __name__ == '__main__':
    main()
