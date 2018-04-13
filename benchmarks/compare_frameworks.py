import click
import numpy as np
import torch

from models.pytorch_model import Model as PTModel
from models.keras_model import Model as KerasModel
from models.tensorflow_model import Model as TFModel
from misc.utils import measure_model


@click.command()
@click.option('-b', '--batch_size', type=int,
              help="Batch size for the sample input.")
def main(batch_size):
    batches = [1, 8, 16, 32, 64]
    if batch_size:
        batches = [batch_size]

    pt_model = PTModel()
    if torch.cuda.is_available():
        pt_model.cuda()
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
