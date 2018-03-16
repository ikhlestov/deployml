#!/bin/bash
cd ../tensorflow
# bazel clean --expunge
# bazel build tensorflow/tools/graph_transforms:transform_graph
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=`pwd`/../deployml/saves/tensorflow/optimized_graph.pb \
    --out_graph=`pwd`/../deployml/saves/tensorflow/quantized_graph.pb  \
    --inputs="input:0" \
    --outputs="output:0" \
    --transforms='quantize_weights'
cd ../deployml
