import os
import sys

BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.insert(0, BASE_DIR)

import tensorflow as tf

from models.tensorflow_model import Model
from constants import TENSORFLOW_SAVES_DIR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(saves_dir=TENSORFLOW_SAVES_DIR):
    os.makedirs(saves_dir, exist_ok=True)
    model = Model()
    saver = tf.train.Saver()
    save_path = os.path.join(os.path.abspath(saves_dir), 'usual_model')
    saver.save(model.sess, save_path)


if __name__ == '__main__':
    main()
