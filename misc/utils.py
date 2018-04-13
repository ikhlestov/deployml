import time

import numpy as np


def measure_model(model, model_name: str, input_batch,
                  n_runs: int = 5, cold_start: int = 2):
    consumptions = []
    for _ in range(cold_start + n_runs):
        start_time = time.time()
        model.predict(input_batch)
        time_cons = time.time() - start_time
        consumptions.append(time_cons)
    consumptions = consumptions[cold_start:]
    results = {
        'model_name': model_name,
        'min_cons': min(consumptions),
        'max_cons': max(consumptions),
        'mean_cons': np.mean(consumptions),
    }
    print("min: {min_cons:.5f}, "
          "mean: {mean_cons:.5f}, "
          "max: {max_cons:.5f} "
          "// {model_name:<11}".format(**results))
