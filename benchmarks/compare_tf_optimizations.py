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


@click.command()
@click.option('-b', '--batch_size', type=int,
              help="Batch size for the sample input.")
def main(batch_size, saves_dir=TENSORFLOW_SAVES_DIR):
    pass


if __name__ == '__main__':
    main()
