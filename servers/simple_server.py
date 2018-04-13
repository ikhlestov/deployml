import json

from aiohttp import web
import numpy as np

from benchmarks.compare_tf_optimizations import BinaryModel
from misc.constants import TENSORFLOW_SAVES_DIR
from models.tensorflow_model import Model


async def index(request):
    # batch = request.data['batch']
    model = request.app["model"]
    batch = request.app["batch"]
    pred = model.predict(batch)
    str_pred = json.dumps(pred.tolist())
    return web.Response(text=str_pred)


def build_app():
    app = web.Application()
    app.router.add_get('/', index)
    model = BinaryModel(
        saves_dir=TENSORFLOW_SAVES_DIR,
        model_file='optimized_graph.pb',
        input_node_name=Model.input_node_name,
        output_node_name=Model.output_node_name
    )
    batch = np.random.random((32, 224, 224, 3))
    app["model"] = model
    app["batch"] = batch
    return app


if __name__ == '__main__':
    app = build_app()
    web.run_app(app, host='0.0.0.0', port=8080)
