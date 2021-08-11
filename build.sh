#!/bin/sh

if [ -d "build" ]
then
      cd build
else
      mkdir build && cd build
fi

cmake -DOpenCV_DIR="$PWD/../opencv/build" \
      -DTensorRT_DIR="/data/quangnd33/TensorRT-7.2.3.4" \
      ..

cmake --build . --config Debug -- -j $(nproc)
