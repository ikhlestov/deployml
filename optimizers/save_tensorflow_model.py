import os

import tensorflow as tf

from misc.constants import TENSORFLOW_SAVES_DIR
from models.tensorflow_model import Model


def main(saves_dir=TENSORFLOW_SAVES_DIR):
    print(saves_dir)
    model = Model()
    saver = tf.train.Saver()
    save_path = os.path.join(saves_dir, 'usual_model')
    saver.save(model.sess, save_path)


if __name__ == '__main__':
    main()
