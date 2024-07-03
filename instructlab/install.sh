#!/usr/bin/bash

source venv/bin/activate
pip cache remove llama_cpp_python
pip install instructlab \
   -C cmake.args="-DLLAMA_CUDA=on" \
   -C cmake.args="-DLLAMA_NATIVE=off"