import json
from concurrent.futures import ThreadPoolExecutor as Executor

from aiohttp import web
import numpy as np

from benchmarks.compare_tf_optimizations import BinaryModel
from misc.constants import TENSORFLOW_SAVES_DIR
from models.tensorflow_model import Model


N_PARALLEL = 4


def task(model, batch):
    pred = model.predict(batch)
    str_pred = json.dumps(pred.tolist())
    return str_pred


async def index(request):
    while True:
        try:
            model = request.app["models"].pop()
            break
        except IndexError:
            pass
    batch = request.app["batch"]
    executor = request.app["executor"]
    future = executor.submit(task, model, batch)
    str_pred = future.result()
    request.app["models"].append(model)
    return web.Response(text=str_pred)


def build_app():
    app = web.Application()
    app.router.add_get('/', index)
    models = []
    for i in range(N_PARALLEL):
        models.append(BinaryModel(
            saves_dir=TENSORFLOW_SAVES_DIR,
            model_file='optimized_graph.pb',
            input_node_name=Model.input_node_name,
            output_node_name=Model.output_node_name
        ))
    batch = np.random.random((32, 224, 224, 3))
    app["models"] = models
    app["batch"] = batch
    app["executor"] = Executor(N_PARALLEL)
    return app


if __name__ == '__main__':
    app = build_app()
    web.run_app(app, host='0.0.0.0', port=8080)
