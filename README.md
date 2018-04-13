## Contents

<!-- MarkdownTOC -->

- [Pre-Requirements](#pre-requirements)
- [Environment SetUp](#environment-setup)
    - [Case 1](#case-1)
    - [Case 1.1](#case-11)
    - [Case 2](#case-2)
    - [Case 3](#case-3)
- [Frameworks comparison](#frameworks-comparison)
- [Tensorflow optimization methods](#tensorflow-optimization-methods)
- [Training optimization approaches](#training-optimization-approaches)
    - [Pruning](#pruning)
    - [XNOR nets](#xnor-nets)
    - [Knowledge distillation](#knowledge-distillation)
- [Simple servers](#simple-servers)
- [Testing](#testing)
    - [Preprocessing and code testing](#preprocessing-and-code-testing)
- [Profiling](#profiling)
- [Routines automation](#routines-automation)
- [Converting weights to the tensorflow](#converting-weights-to-the-tensorflow)
- [Conclusion](#conclusion)

<!-- /MarkdownTOC -->


## Pre-Requirements

- Docker. You may get it [here](https://docs.docker.com/install/)
- Update docker memory limit if necessary
<!-- - Fetch the docker image `docker pull ikhlestov/deployml_dev` -->
- Git, Python>=3.5


## Environment SetUp

### Case 1

- Clone the workshop repository 

    `git clone git@github.com:ikhlestov/deployml.git && cd deployml`

- Create virtualenv

    `python3.6 -m venv .venv && source .venv/bin/activate`

- Install corresponding requirements 

    `pip install -r requirements/dev_mac.txt`

    or

    `pip install -r requirements/dev_ubuntu_cpu.txt`

**Note**: requirements are based on python3.6. If you have another python version you should change link to the pytorch wheel at the requirements file which you may get [here](http://pytorch.org/)

### Case 1.1

Additionally download tensorflow source code nearby:

`git clone https://github.com/tensorflow/tensorflow.git -b v1.6.0`

### Case 2

pull small docker container:

`docker pull ikhlestov/deployml_dev_small`

or

pull large docker container(in case of really good Internet connection):

`docker pull ikhlestov/deployml_dev`

### Case 3

Build your own docker container:

- Clone the workshop repository `git clone git@github.com:ikhlestov/deployml.git && cd deployml`
- Check dockers containers defined at the *dockers* folder
- Run build commands:
    - `docker build -f dockers/Dev . -t ikhlestov/deployml_dev` (for the workshop you should build only this image)
    - `docker build -f dockers/Dev_small . -t ikhlestov/deployml_dev_small`
    - `docker build -f dockers/Prod . -t ikhlestov/deployml_prod`
- Compare their sizes `docker images | grep "deployml_dev\|deployml_dev_small\|deployml_prod"`

Notes:

- Don't forget about [.dockerignore file](https://docs.docker.com/engine/reference/builder/#dockerignore-file).
- Try to organize your docker files to use cache.
- Optimize your docker containers
- Try to release with some smaller distributions.
- You may use [multistage builds](https://docs.docker.com/develop/develop-images/multistage-build/)

## Frameworks comparison

- Check defined models in the [models](models) folder
- Run docker container with mounted directory:
    
    `docker run -v $(pwd):/deployml -p 6060:6060 -p 8080:8080 -it ikhlestov/deployml_dev /bin/bash`

- Run time measurements inside docker:

    `python benchmarks/compare_frameworks.py`

## Tensorflow optimization methods

1. Save our tensorflow model.

    `python optimizers/save_tensorflow_model.py`

    1.1 Import saved model to tensorboard

    `python misc/import_pb_to_tensorboard.py --model_dir saves/tensorflow/usual_model.pbtxt --log_dir saves/tensorboard/usual_model --graph_type PbTxt`

    1.2 Run tensorboard in the background

    `tensorboard --logdir saves/tensorboard --port 6060 --host=0.0.0.0 &`

2. Build frozen graph. More about it you may read [here](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc)

    `python optimizers/get_frozen_graph.py`

    `python misc/import_pb_to_tensorboard.py  --model_dir saves/tensorflow/constant_graph.pb --log_dir saves/tensorboard/constant_graph
    `

3. Build optimized frozen graph

    `python optimizers/get_optimized_frozen_graph.py`

    `python misc/import_pb_to_tensorboard.py --model_dir saves/tensorflow/optimized_graph.pb --log_dir saves/tensorboard/optimized_graph`

4. Get quantized graph:

    3.1 With plain python ([link to script](https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/tools/quantization/quantize_graph.py))
    
        python /tensorflow/tensorflow/tools/quantization/quantize_graph.py \
            --input=saves/tensorflow/optimized_graph.pb \
            --output=saves/tensorflow/quantized_graph_python.pb \
            --output_node_names="output" \
            --mode=weights

    3.2. With bazel ([tensorflow tutorial](https://www.tensorflow.org/performance/quantization))

        ../tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
            --in_graph=`pwd`/saves/tensorflow/optimized_graph.pb \
            --out_graph=`pwd`/saves/tensorflow/quantized_graph_bazel.pb  \
            --inputs="input:0" \
            --outputs="output:0" \
            --transforms='quantize_weights'

    3.3 Note: [tf.contrib.quantize](https://www.tensorflow.org/api_docs/python/tf/contrib/quantize) provide only simulated quantization.

    3.4 Import quantized models to the tensorboard

        python misc/import_pb_to_tensorboard.py \
            --model_dir saves/tensorflow/quantized_graph_bazel.pb \
            --log_dir saves/tensorboard/quantized_graph_bazel
        
        python misc/import_pb_to_tensorboard.py \
            --model_dir saves/tensorflow/quantized_graph_python.pb \
            --log_dir saves/tensorboard/quantized_graph_python

5. Compare resulted graphs

    5.1 sizes `ls -l saves/tensorflow/`

    5.2 architecture at the [tensorboard](http://127.0.0.1:6006)

    5.3 Compare resulted graphs performance `python benchmarks/compare_tf_optimizations.py`


6. Try various restrictions

    6.1 CPU restriction

        docker run -v $(pwd):/deployml -it --cpus="1.0" ikhlestov/deployml_dev /bin/bash

    6.2 Memory restriction

        docker run -v $(pwd):/deployml -it --memory=1g ikhlestov/deployml_dev /bin/bash

    6.3 Use GPUs

        docker run --runtime=nvidia -v $(pwd):/deployml -it ikhlestov/deployml_dev /bin/bash

    6.3 Try to run two models on two different CPUs
    6.4 Try to run two models on two CPU simultaneously


<!-- In case you want to run this code locally you should:

- install bazel ([manual](https://docs.bazel.build/versions/master/install.html))
- clone tensorflow repo `git clone https://github.com/tensorflow/tensorflow.git && cd tensorflow`
- build required script `bazel build tensorflow/tools/graph_transforms:transform_graph`
 -->


## Training optimization approaches

You may also take a look at other methods ([list of resources](optimization_approaches.md)) like:

### Pruning

![Pruning](/images/02_pruning.jpg)

### XNOR nets

![XNOR nets](/images/03_xnor_net.png)

### Knowledge distillation

![Knowledge distillation](/images/04_distillation.png)


## Simple servers

**One-to-one server([servers/simple_server.py](servers/simple_server.py))**

![One-to-one server](/images/05_simplest_BE_architecture.png)

**Scaling with multiprocessing([servers/processes_server.py](servers/processes_server.py))**

![Scaling with multiprocessing](/images/06_multiprocessing_scale.png)

You may start servers (not simultaneously) as:

    python servers/simple_server.py
   
or 

    python servers/processes_server.py

and test them with:
   
    python servers/tester.py


**Queues based(Kafka, RabbitMQ, etc)**

![Queues based](/images/07_messages_based.png)

**Serving with [tf-serving](https://www.tensorflow.org/serving/)**


## Testing

### Preprocessing and code testing

Q: Where data preprocessing should be done? CPU or GPU or even another host?

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
- Deterministic output
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
