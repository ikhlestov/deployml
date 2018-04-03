import os
import sys

BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.insert(0, BASE_DIR)

import click
import numpy as np

from models.pytorch_model import Model as PTModel
from models.keras_model import Model as KerasModel
from models.tensorflow_model import Model as TFModel
from utils import measure_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@click.command()
@click.option('-b', '--batch_size', type=int,
              help="Batch size for the sample input.")
def main(batch_size):
    batches = [1, 8, 16, 32, 64]
    if batch_size:
        batches = [batch_size]

    pt_model = PTModel()
    keras_model = KerasModel()
    tf_model = TFModel()
    for batch_size in batches:
        print("Batch size: {}".format(batch_size))
        pt_batch = np.random.random((batch_size, 3, 224, 224))
        measure_model(pt_model, "Pytorch", pt_batch)
        tf_batch = np.random.random((batch_size, 224, 224, 3))
        measure_model(keras_model, "Keras", tf_batch)
        measure_model(tf_model, "Tensorflow", tf_batch)


if __name__ == '__main__':
    main()
