import os

import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.core.framework import graph_pb2

from misc.constants import TENSORFLOW_SAVES_DIR
from models.tensorflow_model import Model


def main(saves_dir=TENSORFLOW_SAVES_DIR):
    # load previously frozen graph
    frozen_save_path = os.path.join(saves_dir, 'constant_graph.pb')
    input_graph_def = graph_pb2.GraphDef()
    with tf.gfile.Open(frozen_save_path, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    # optimize graph
    optimized_constant_graph = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        [Model.input_node_name],
        [Model.output_node_name],
        Model.input_data_type.as_datatype_enum
    )

    # save optimized graph
    optimized_graph_path = os.path.join(saves_dir, 'optimized_graph.pb')
    with tf.gfile.GFile(optimized_graph_path, "wb") as f:
        f.write(optimized_constant_graph.SerializeToString())


if __name__ == '__main__':
    main()
