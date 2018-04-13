# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ================================
"""Imports a protobuf model as a graph in Tensorboard."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saved_model_pb2

from tensorflow.python.util import compat
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary


def import_to_tensorboard(model_dir, log_dir):
  """View an imported protobuf model (`.pb` file) as a graph in Tensorboard.

  Args:
    model_dir: The location of the protobuf (`pb`) model to visualize
    log_dir: The location for the Tensorboard log to begin visualization from.

  Usage:
    Call this function with your model location and desired log directory.
    Launch Tensorboard by pointing it to the log directory.
    View your imported `.pb` model as a graph.
  """
  with session.Session(graph=ops.Graph()) as sess:
    with gfile.FastGFile(model_dir, "rb") as f:
      graph_def = graph_pb2.GraphDef()
      graph_def.ParseFromString(f.read())
      importer.import_graph_def(graph_def)

    pb_visual_writer = summary.FileWriter(log_dir)
    pb_visual_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(log_dir))


def import_saved_model_to_tensorboard(model_dir, log_dir):
  with session.Session() as sess:
    with gfile.FastGFile(model_dir, 'rb') as f:

        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
        if 1 != len(sm.meta_graphs):
          print('More than one graph found. Not sure which to write')
          sys.exit(1)

        g_in = importer.import_graph_def(sm.meta_graphs[0].graph_def)
  pb_visual_writer = summary.FileWriter(log_dir)
  pb_visual_writer.add_graph(sess.graph)
  print("Model Imported. Visualize by running: "
        "tensorboard --logdir={}".format(log_dir))


def import_pbtxt_model_to_tensorboard(model_dir, log_dir):
  with session.Session() as sess:
    with gfile.FastGFile(model_dir, 'rb') as f:
        graph_def = graph_pb2.GraphDef()
        data = compat.as_bytes(f.read())
        text_format.Merge(data, graph_def)
        importer.import_graph_def(graph_def, name='model_pbtxt')
    pb_visual_writer = summary.FileWriter(log_dir)
    pb_visual_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(log_dir))


def main(unused_args):
  if FLAGS.graph_type == 'SavedModel':
    import_saved_model_to_tensorboard(FLAGS.model_dir, FLAGS.log_dir)
  
  if FLAGS.graph_type == 'GraphDef':
    import_to_tensorboard(FLAGS.model_dir, FLAGS.log_dir)

  if FLAGS.graph_type == 'PbTxt':
    import_pbtxt_model_to_tensorboard(FLAGS.model_dir, FLAGS.log_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      required=True,
      help="The location of the protobuf (\'pb\') model to visualize.")
  parser.add_argument(
      "--log_dir",
      type=str,
      default="",
      required=True,
      help="The location for the Tensorboard log to begin visualization from.")
  parser.add_argument(
      "--graph_type",
      type=str,
      default="GraphDef",
      help="SavedModel, GraphDef, or PbTxt")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
