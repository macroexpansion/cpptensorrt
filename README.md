# TensorRT C++

### Requirements
- TensorRT 7.2.3.4
- gcc 7.5.0
- cmake 3.10.2
- CUDA 11.1.1
- cuDNN 8.1.1
- [openCV 4.5.1](https://docs.opencv.org/4.5.1/d7/d9f/tutorial_linux_install.html)

### Build
- change `DOpenCV_DIR` and `DTensorRT_DIR` to your OpenCV and TensorRT installation in `build.sh` and run `sh build.sh`

### Run
(required export PyTorch model to ONNX model first, this repo only convert from ONNX to TensorRT)
- run `./bin/serialize` to build tensorRT engine from ONNX model and serialize it
- run `./bin/inference` to run inference
