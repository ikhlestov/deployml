import os
import sys

BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.insert(0, BASE_DIR)

import keras
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph

from constants import KERAS_SAVES_DIR


def main():
    load_path = os.path.join(KERAS_SAVES_DIR, 'keras_model.h5')
    save_path = os.path.join(KERAS_SAVES_DIR, 'exported')

    keras.backend.set_learning_phase(0)
    model = keras.models.load_model(load_path)

    model_input = model.input.name.replace(':0', '')
    model_output = model.output.name.replace(':0', '')
    print("model input:", model_input)
    print("model output:", model_output)

    session = keras.backend.get_session()

    # graph_def = session.graph.as_graph_def()
    tf.train.Saver().save(session, save_path + '.ckpt')
    tf.train.write_graph(
        session.graph.as_graph_def(),
        logdir='.',
        name=save_path + '.binary.pb',
        as_text=False
    )


if __name__ == '__main__':
    main()
