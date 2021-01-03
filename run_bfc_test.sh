#!/bin/bash

bin_dir=${HOME}/workspace/Github/tensorflow
trace_dir=${HOME}/workspace/Github/SimGraph
# bin_dir=${HOME}/vfonel/tensorflow-1.15.2
run_cmd=${bin_dir}/bazel-bin/tensorflow/core/bfc_allocator_test

# model=$1
# batch_size=$2
# export TF_BFC_ALLOCATOR_TEST_TRACE_FILE=${HOME}/vfonel/SimGraph/graph_p100/${model}/train_${model}_${batch_size}_dup2_alloc_trace.log
# ${run_cmd}

# models="alexnet vgg16 resnet50 resnet152 inception3 inception4 Bert"
# models="alexnet vgg16 resnet50 inception3 Bert"
models="deepspeech2"

for model in $models
do
  if [ $model ==  "alexnet" ];then
    batch_size=512
  elif [ $model == "Bert" ];then
    batch_size=32
  elif [ $model == "deepspeech2" ];then
    batch_size=32
  elif [ $model == "Tacotron" ];then
    batch_size=32
  else
    batch_size=64
  fi
  # export TF_BFC_ALLOCATOR_TEST_TRACE_FILE=${trace_dir}/graph_p100/${model}/train_${model}_${batch_size}_dup2_alloc_trace.log
  export TF_BFC_ALLOCATOR_TEST_TRACE_FILE=${trace_dir}/graph_p100/${model}/train_${model}_${batch_size}_alloc_trace.log
  echo "Run $model BFCAllocatorTest"
  ${run_cmd} > ./logs/${model}_${batch_size}_alloc.log 2>&1
  # ${run_cmd} > ./logs/${model}_${batch_size}_dup2_alloc.log 2>&1
done