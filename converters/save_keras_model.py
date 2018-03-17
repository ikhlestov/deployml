import os
import sys

BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.insert(0, BASE_DIR)

from constants import KERAS_SAVES_DIR
from models.keras_model import Model


def main():
    model = Model()
    save_path = os.path.join(KERAS_SAVES_DIR, 'keras_model.h5')
    model.model.save(save_path)


if __name__ == '__main__':
    main()
