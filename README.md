## Requirements

In any case:

- Docker. You may get it [here](https://docs.docker.com/install/).
- Git
- Clone the repo `git clone git@github.com:ikhlestov/deployml.git`

For local setup you will also need:

- Python 3.6
- Bazel ([installation manual](https://docs.bazel.build/versions/master/install.html))
- Tensorflow source code `git clone https://github.com/tensorflow/tensorflow.git`

## Local SetUp

1. Create virtual env `python3.6 -m venv .venv && source .venv/bin/activate`
2. Install required packages according to your OS:

    - Ubuntu - `pip install -r requirements/local_ubuntu.txt`
    - Mac - `pip install -r requirements/local_mac.txt`

## Docker SetUp

- Dev container `docker build -f dockers/Dev . -t workshop_dev`
- Dev container `docker build -f dockers/ProdLarge . -t workshop_prod_large`
- Dev container `docker build -f dockers/ProdSmall . -t workshop_prod_small`
- Compare their sizes `docker images | grep "workshop_dev\|workshop_prod_large\|workshop_prod_small"`

<!---
Don't forget about `.dockerignore` file.
Try to organize you docker file to use cache as much as possible.
Develop with ubuntu, release with some smaller distros.
-->

## Start with various frameworks

- Check defined models in the `models` folder
- Run docker container with mounted directory `docker run -v `pwd`:/project -it workshop_dev /bin/bash`
- Run time measurements inside docker `cd /project && python compare_frameworks.py`
- Run time measurements for every model locally `python compare_frameworks.py`(if you've passed local setup)

## Optimize tensorflow

- Build frozen graph with `python get_frozen_model.py`. More about it you may read [here](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc)
- Compare frozen graph with usual one `python compare_frozen_graph.py`
- Build optimized frozen graph `python get_optimized_frozen_graph.py`
- Check what speedup we've got `python compare_optimized_graph.py`

<!---
The main usability that you may ship your model as one binary file.
You should freeze/optimize/serve model with the same tensorflow version.
Quantization may reduce model size, but also decrease model speed.
-->
