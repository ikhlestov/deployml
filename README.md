<!-- MarkdownTOC -->

- [Pre-Requirements](#pre-requirements)
- [Environment SetUp](#environment-setup)
- [Frameworks comparison](#frameworks-comparison)
- [Tensorflow optimization methods](#tensorflow-optimization-methods)
- [Training optimization approaches](#training-optimization-approaches)
- [Simple servers](#simple-servers)
- [Testing](#testing)
    - [Load testing with various restrictions](#load-testing-with-various-restrictions)
    - [Preprocessing and code testing](#preprocessing-and-code-testing)
- [Profiling](#profiling)
- [Routines automation](#routines-automation)
- [Converting weights to the tensorflow](#converting-weights-to-the-tensorflow)
- [Conclusion](#conclusion)

<!-- /MarkdownTOC -->


## Pre-Requirements

- Docker. You may get it [here](https://docs.docker.com/install/)
- Update docker memory limit if necessary
- Fetch the docker image `docker pull ikhlestov/deployml_dev`
- Git
- Clone the workshop repository `git clone git@github.com:ikhlestov/deployml.git`


## Environment SetUp

- Check dockers containers defined at the `dockers` folder
- You may try to build your own containers:
    - `docker build -f dockers/Dev . -t ikhlestov/deployml_dev`
    - `docker build -f dockers/Pro . -t ikhlestov/deployml_prod`
- Compare their sizes `docker images | grep "deployml_dev\|deployml_prod"`

Notes:

- Don't forget about [.dockerignore file](https://docs.docker.com/engine/reference/builder/#dockerignore-file).
- Try to organize your docker files to use cache.
- Optimize your docker containers
- Try to release with some smaller distributions.
- You may use [multistage builds](https://docs.docker.com/develop/develop-images/multistage-build/)

## Frameworks comparison

- Check defined models in the [models](models) folder
- Run docker container with mounted directory:
    
    `docker run -v $(pwd):/deployml -it ikhlestov/deployml_dev /bin/bash`

- Run time measurements inside docker:

    `cd /deployml && python benchmarks/compare_frameworks.py`

- Optional:
    
    - setup local environment `python3.6 -m venv .venv && source .venv/bin/activate && pip install -r requirements/local_mac.txt`
    - Run time measurements for every model locally `python benchmarks/compare_frameworks.py`


## Tensorflow optimization methods

1. Save our tensorflow model.

    `python optimizers/save_tensorflow_model.py`

2. Build frozen graph. More about it you may read [here](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc)

    `python optimizers/get_frozen_graph.py`

3. Build optimized frozen graph
    
    `python optimizers/get_optimized_frozen_graph.py`

4. Get quantized graph:
    
    3.1. With bazel ([tensorflow tutorial](https://www.tensorflow.org/performance/quantization))

        ../tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
            --in_graph=`pwd`/saves/tensorflow/optimized_graph.pb \
            --out_graph=`pwd`/saves/tensorflow/quantized_graph_bazel.pb  \
            --inputs="input:0" \
            --outputs="output:0" \
            --transforms='quantize_weights'


    3.2 With plain python ([link to script](https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/tools/quantization/quantize_graph.py))
    
        python /tensorflow/tensorflow/tools/quantization/quantize_graph.py \
            --input=saves/tensorflow/optimized_graph.pb \
            --output=saves/tensorflow/quantized_graph_python.pb \
            --output_node_names="output" \
            --mode=weights

    3.3 Note: [tf.contrib.quantize](https://www.tensorflow.org/api_docs/python/tf/contrib/quantize) provide only simulated quantization.

5. Compare resulted graphs `python benchmarks/compare_tf_optimizations.py`

In case you want to run this code locally you should:

- install bazel ([manual](https://docs.bazel.build/versions/master/install.html))
- clone tensorflow repo `git clone https://github.com/tensorflow/tensorflow.git && cd tensorflow`
- build required script `bazel build tensorflow/tools/graph_transforms:transform_graph`


## Training optimization approaches

You may also take a look at other methods ([list of resources](resources.md)) like:

- More about pruning
- Quantization
- XNOR nets
- Knowledge distillation


## Simple servers

- One-to-one server
- Scaling with multiprocessing
- Queues based(Kafka, RabbitMQ, etc)
- Serving with [tf-serving](https://www.tensorflow.org/serving/)


## Testing

### Load testing with various restrictions

- CPU restriction `docker run -v $(pwd):/deployml -it --cpus="1.0" ikhlestov/deployml_dev /bin/bash`
- Memory restriction `docker run -v $(pwd):/deployml -it --memory=1g ikhlestov/deployml_dev /bin/bash`
- Try to run two models on two different CPUs
- Try to run two models on two CPU simultaneously

<!---
Where data preprocessing should be done? CPU or GPU or even another host?
-->

### Preprocessing and code testing

Q: Where is preprocessing should be done - on CPU or GPU?

- enter to the preprocessing directory `cd preprocessing`
- run various resizers benchmarks `python benchmark.py`
    
    - Note: opencv may be installed from PyPi for python3

- check unified resizer at the `image_preproc.py`
- try to run tests for it `pytest test_preproc.py`(and they will fail)
- fix resizer
- run tests again `pytest test_preproc.py`

What else should be tested(really - as much as possible):

- General network inference
- Model loading/saving
- New models deploy
- Any preprocessing
- Corrupted inputs - Nan, Inf, zeros
- Determinism
- Input ranges/distributions
- Output ranges/distributions
- Test that model will fail in known cases
- ...
- Just check [this video](https://youtu.be/T_YWBGApUgs?t=5h59m40s) :)

You my run tests:

- At the various docker containers
- Under the [tox](https://tox.readthedocs.io/en/latest/)


## Profiling

- Code:
    
    - [Line profiler](https://github.com/rkern/line_profiler)
    - [CProfile](https://docs.python.org/3.6/library/profile.html)
    - [PyCharm Profiler](https://www.jetbrains.com/help/pycharm/optimizing-your-code-using-profilers.html)

- Tensorflow:

    - [Chrome trace format based](https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d)
    - [Tf benchmark tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/benchmark)
    - [Tf benchmark class](https://www.tensorflow.org/api_docs/python/tf/test/Benchmark)

- CPU/GPU:

    - nvidia-smi
    - [gpustat](https://github.com/wookayin/gpustat)
    - [psutil](https://pypi.python.org/pypi/psutil)
    - [nvidia profiler](https://developer.nvidia.com/nvidia-visual-profiler)

- Lifetime benchmark - [airspeed velocity](https://github.com/airspeed-velocity/asv)


## Routines automation

- Continuous integration:

    - Jenkins
    - Travis
    - TeamCity
    - CircleCI

- Clusters:

    - Kubernetes
    - Mesos
    - Docker swarm

- Configuration management:

    - Terraform
    - Ansible
    - Chef
    - Puppet
    - SaltStack


## Converting weights to the tensorflow

- Converting from keras to tensorflow:

    - Get keras saved model `python converters/save_keras_model.py`
    - Convert keras model to the tensorflow save format `python converters/convert_keras_to_tf.py`

- Converting from PyTorch to tensorflow:

    - Trough keras - [converter](https://github.com/nerox8664/pytorch2keras)
    - Manually

In any case you should know about:

- [Open Neural Network Exchange](https://github.com/onnx/onnx)
- [Deep Learning Model Convertors](https://github.com/ysh329/deep-learning-model-convertor)


## Conclusion

I'm grateful for the cool ideas to Alexandr Onbysh, Aleksandr Obednikov, Kyryl Truskovskyi and to the Ring Urkaine in overall.

Take a look at the [checklist](checklist.md).

Thank you for reading!
